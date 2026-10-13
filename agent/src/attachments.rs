use std::path::{Path, PathBuf};

use base64::prelude::*;
use sha3::{Digest, Sha3_256};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AttachError {
    #[error("denied: {0}")]
    Denied(String),
    #[error("failed: {0}")]
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Attachment {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "mimeType", alias = "mime_type")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromptInput {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub attachment: Option<Attachment>,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
}

#[must_use]
pub fn select_attachments_to_save(prompt: &PromptInput) -> Vec<&Attachment> {
    let list: &[Attachment] = if !prompt.attachments.is_empty() {
        &prompt.attachments
    } else if let Some(ref att) = prompt.attachment {
        std::slice::from_ref(att)
    } else {
        &[]
    };
    list.iter().filter(|att| !att.mime_type.starts_with("image/")).collect()
}

#[must_use]
pub fn build_run_prompt(input: &PromptInput) -> Vec<serde_json::Value> {
    let mut parts: Vec<serde_json::Value> = Vec::new();
    if !input.text.is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": input.text,
        }));
    }

    let mut process_attachment = |att: &Attachment| {
        let is_image = att.mime_type.starts_with("image/");
        if is_image {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", att.mime_type, att.data),
                }
            }));
            if let Some(name) = att.name.as_deref().filter(|n| !n.is_empty()) {
                parts.push(serde_json::json!({
                    "type": "text",
                    "text": format!("<system-reminder>\nAttached image \"{name}\" is shown inline.\n</system-reminder>"),
                }));
            }
        } else if let Some(name) = att.name.as_deref().filter(|n| !n.is_empty()) {
            parts.push(serde_json::json!({
                "type": "text",
                "text": format!("<system-reminder>\nAttached file \"{name}\" has been saved to the workspace. You can read, extract, or examine it using your tools.\n</system-reminder>"),
            }));
        }
    };

    if let Some(ref att) = input.attachment {
        process_attachment(att);
    }

    if !input.attachments.is_empty() {
        let is_first_processed = input.attachment.is_some() && input.attachment.as_ref() == input.attachments.first();
        let skip_count = usize::from(is_first_processed);
        for att in input.attachments.iter().skip(skip_count) {
            process_attachment(att);
        }
    }

    if parts.is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": "",
        }));
    }

    parts
}

fn resolve_attachment_target(root: &Path, name: &str) -> Result<(PathBuf, PathBuf), AttachError> {
    let canonical_root = root
        .canonicalize()
        .map_err(|e| AttachError::Failed(format!("sandbox root not found: {e}")))?;

    let req_path = Path::new(name);
    let rel_components: Vec<_> = if req_path.is_absolute() {
        match req_path.strip_prefix(&canonical_root) {
            Ok(rel) => rel.components().collect(),
            Err(_) => {
                return Err(AttachError::Denied(format!("absolute path outside sandbox: {name}")));
            }
        }
    } else {
        req_path.components().collect()
    };

    let mut target = canonical_root.clone();
    for comp in rel_components {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if target == canonical_root {
                    return Err(AttachError::Denied(format!("path traversal out of sandbox: {name}")));
                }
                target.pop();
            }
            std::path::Component::Normal(c) => {
                target.push(c);
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(AttachError::Denied(format!("path traversal out of sandbox: {name}")));
            }
        }
    }

    if !target.starts_with(&canonical_root) {
        return Err(AttachError::Denied(format!("path escapes sandbox: {name}")));
    }

    if target.exists() {
        let canonical_target = target.canonicalize().map_err(|e| AttachError::Failed(e.to_string()))?;
        if !canonical_target.starts_with(&canonical_root) {
            return Err(AttachError::Denied("symlink escapes sandbox".to_string()));
        }
    } else {
        let mut ancestor = target.as_path();
        while let Some(parent) = ancestor.parent() {
            if parent.exists() {
                let canonical_parent = parent.canonicalize().map_err(|e| AttachError::Failed(e.to_string()))?;
                if !canonical_parent.starts_with(&canonical_root) {
                    return Err(AttachError::Denied("symlink escapes sandbox".to_string()));
                }
                break;
            }
            ancestor = parent;
        }
    }

    Ok((canonical_root, target))
}

fn resolve_collision_path(file_path: &Path, buffer: &[u8], canonical_root: &Path) -> PathBuf {
    let file_name = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let (origin, ext) = match file_name.rfind('.') {
        Some(idx) if idx > 0 => (&file_name[..idx], &file_name[idx..]),
        _ => (file_name, ""),
    };

    let mut hasher = Sha3_256::new();
    hasher.update(buffer);
    let hash_bytes = hasher.finalize();
    let hash_hex = format!("{hash_bytes:x}");
    let short_hash = format!("{}{}", &hash_hex[..4], &hash_hex[hash_hex.len() - 1..]);

    let new_file_name = format!("{origin}_{short_hash}{ext}");
    let parent_dir = file_path.parent().unwrap_or(canonical_root);
    parent_dir.join(new_file_name)
}

/// Save a non-image attachment into the sandbox root directory.
///
/// # Errors
///
/// Returns `AttachError::Denied` if the path escapes the sandbox.
/// Returns `AttachError::Failed` if reading or writing files fails.
pub async fn save_attachment(root: &Path, att: &Attachment) -> Result<Option<String>, AttachError> {
    let Some(name) = &att.name else {
        return Ok(None);
    };
    if name.is_empty() || att.data.is_empty() {
        return Ok(None);
    }

    let (canonical_root, target) = resolve_attachment_target(root, name)?;

    let clean_data: String = att.data.chars().filter(|c| !c.is_whitespace()).collect();
    let buffer = BASE64_STANDARD
        .decode(&clean_data)
        .map_err(|e| AttachError::Failed(format!("invalid base64: {e}")))?;

    let mut file_path = target;
    if file_path.exists() {
        let existing = tokio::fs::read(&file_path)
            .await
            .map_err(|e| AttachError::Failed(e.to_string()))?;
        if existing == buffer {
            let rel = file_path
                .strip_prefix(&canonical_root)
                .map_err(|e| AttachError::Failed(e.to_string()))?
                .to_string_lossy()
                .replace('\\', "/");
            return Ok(Some(rel));
        }

        file_path = resolve_collision_path(&file_path, &buffer, &canonical_root);
    }

    if let Some(parent) = file_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| AttachError::Failed(e.to_string()))?;
    }

    tokio::fs::write(&file_path, &buffer)
        .await
        .map_err(|e| AttachError::Failed(e.to_string()))?;

    let rel = file_path
        .strip_prefix(&canonical_root)
        .map_err(|e| AttachError::Failed(e.to_string()))?
        .to_string_lossy()
        .replace('\\', "/");
    Ok(Some(rel))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(name: &str) -> Attachment {
        Attachment {
            name: Some(name.into()),
            mime_type: "image/png".into(),
            data: "iVBORw0KGgoAAAANSUhEUg==".into(),
        }
    }

    fn file(name: &str) -> Attachment {
        Attachment {
            name: Some(name.into()),
            mime_type: "application/zip".into(),
            data: "UEsDBA==".into(),
        }
    }

    #[test]
    fn test_non_images_saved_images_inline() {
        let prompt = PromptInput {
            text: "hi".into(),
            attachment: None,
            attachments: vec![img("a.png"), file("b.zip")],
        };
        let save = select_attachments_to_save(&prompt);
        assert_eq!(save.len(), 1);
        assert_eq!(save[0].name.as_deref(), Some("b.zip"));
    }

    #[test]
    fn test_select_attachments_prefers_plural() {
        let prompt = PromptInput {
            text: "test".into(),
            attachment: Some(file("single.zip")),
            attachments: vec![file("multi.tar")],
        };
        let save = select_attachments_to_save(&prompt);
        assert_eq!(save.len(), 1);
        assert_eq!(save[0].name.as_deref(), Some("multi.tar"));
    }

    #[test]
    fn test_select_attachments_fallback_to_singular() {
        let prompt = PromptInput {
            text: "test".into(),
            attachment: Some(file("single.zip")),
            attachments: vec![],
        };
        let save = select_attachments_to_save(&prompt);
        assert_eq!(save.len(), 1);
        assert_eq!(save[0].name.as_deref(), Some("single.zip"));
    }

    #[test]
    fn test_build_run_prompt_dedup_and_inline() {
        let single = img("a.png");
        let prompt = PromptInput {
            text: "look".into(),
            attachment: Some(single.clone()),
            attachments: vec![single, file("doc.pdf")],
        };
        let parts = build_run_prompt(&prompt);
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "look");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        );
        assert_eq!(parts[2]["type"], "text");
        assert!(
            parts[2]["text"]
                .as_str()
                .unwrap()
                .contains("Attached image \"a.png\" is shown inline.")
        );
        assert_eq!(parts[3]["type"], "text");
        assert!(
            parts[3]["text"]
                .as_str()
                .unwrap()
                .contains("Attached file \"doc.pdf\" has been saved to the workspace.")
        );
    }

    #[test]
    fn test_build_run_prompt_empty_fallback() {
        let prompt = PromptInput::default();
        let parts = build_run_prompt(&prompt);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "");
    }

    #[tokio::test]
    async fn test_save_attachment_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let att = Attachment {
            name: Some("../../x.png".into()),
            mime_type: "image/png".into(),
            data: "iVBORw==".into(),
        };
        let res = save_attachment(dir.path(), &att).await;
        assert!(matches!(res, Err(AttachError::Denied(_))));
    }

    #[tokio::test]
    async fn test_save_attachment_absolute_escape_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let att = Attachment {
            name: Some("/etc/passwd".into()),
            mime_type: "text/plain".into(),
            data: "aGVsbG8=".into(),
        };
        let res = save_attachment(dir.path(), &att).await;
        assert!(matches!(res, Err(AttachError::Denied(_))));
    }

    #[tokio::test]
    async fn test_save_attachment_empty_data() {
        let dir = tempfile::tempdir().unwrap();
        let att = Attachment {
            name: Some("empty.txt".into()),
            mime_type: "text/plain".into(),
            data: String::new(),
        };
        let res = save_attachment(dir.path(), &att).await.unwrap();
        assert_eq!(res, None);
        assert!(!dir.path().join("empty.txt").exists());
    }

    #[tokio::test]
    async fn test_save_attachment_no_name() {
        let dir = tempfile::tempdir().unwrap();
        let att = Attachment {
            name: None,
            mime_type: "text/plain".into(),
            data: "aGVsbG8=".into(),
        };
        let res = save_attachment(dir.path(), &att).await.unwrap();
        assert_eq!(res, None);
    }

    #[tokio::test]
    async fn test_save_attachment_success_and_identical_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let att = Attachment {
            name: Some("sub/hello.txt".into()),
            mime_type: "text/plain".into(),
            data: "aGVsbG8=".into(),
        };
        let res = save_attachment(dir.path(), &att).await.unwrap();
        assert_eq!(res, Some("sub/hello.txt".into()));
        let content = std::fs::read_to_string(dir.path().join("sub/hello.txt")).unwrap();
        assert_eq!(content, "hello");

        let res2 = save_attachment(dir.path(), &att).await.unwrap();
        assert_eq!(res2, Some("sub/hello.txt".into()));
    }

    #[tokio::test]
    async fn test_save_attachment_collision_sha3_rename() {
        let dir = tempfile::tempdir().unwrap();
        let att1 = Attachment {
            name: Some("data.bin".into()),
            mime_type: "application/octet-stream".into(),
            data: "AQID".into(),
        };
        let saved1 = save_attachment(dir.path(), &att1).await.unwrap();
        assert_eq!(saved1, Some("data.bin".into()));

        let att2 = Attachment {
            name: Some("data.bin".into()),
            mime_type: "application/octet-stream".into(),
            data: "BAUG".into(),
        };
        let saved2 = save_attachment(dir.path(), &att2).await.unwrap().unwrap();
        assert_ne!(saved2, "data.bin");
        assert!(saved2.starts_with("data_"));
        assert!(
            std::path::Path::new(&saved2)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
        );

        let mut hasher = Sha3_256::new();
        hasher.update([4, 5, 6]);
        let hash_hex = format!("{:x}", hasher.finalize());
        let expected_hash = format!("{}{}", &hash_hex[..4], &hash_hex[hash_hex.len() - 1..]);
        assert_eq!(saved2, format!("data_{expected_hash}.bin"));

        assert_eq!(std::fs::read(dir.path().join("data.bin")).unwrap(), vec![1, 2, 3]);
        assert_eq!(std::fs::read(dir.path().join(&saved2)).unwrap(), vec![4, 5, 6]);
    }
}
