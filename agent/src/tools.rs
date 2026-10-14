// https://github.com/af-t/agent-sdk/tree/v4.0.0/src/tools

use base64::Engine as _;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolError {
    #[error("denied: {0}")]
    Denied(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("failed: {0}")]
    Failed(String),
    #[error("too large: {0}")]
    TooLarge(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ToolCtx {
    pub sandbox_root: PathBuf,
    pub http: reqwest::Client,
    pub shell_timeout_secs: u64,
    pub fetch_timeout_secs: u64,
    pub fetch_max_bytes: usize,
}

impl Default for ToolCtx {
    fn default() -> Self {
        Self {
            sandbox_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            http: reqwest::Client::new(),
            shell_timeout_secs: 120,
            fetch_timeout_secs: 30,
            fetch_max_bytes: 2 * 1024 * 1024,
        }
    }
}

#[cfg(test)]
impl ToolCtx {
    #[must_use]
    pub fn test_in(dir: &tempfile::TempDir) -> Self {
        Self {
            sandbox_root: dir.path().to_path_buf(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
#[must_use]
pub fn test_in(dir: &tempfile::TempDir) -> ToolCtx {
    ToolCtx::test_in(dir)
}

#[must_use]
#[allow(clippy::too_many_lines)]
pub fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "readFile",
            description: "Read a file. Text and notebooks include line numbers and support pagination. Images, PDFs, audio, and video are returned as multimodal content when supported. Other binary files return metadata and a hex preview.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "description": "File to read." },
                    "startLine": { "type": "number", "description": "First line to return." },
                    "endLine": { "type": "number", "description": "Last line to return." },
                    "maxLines": { "type": "number", "description": "Maximum lines to return. The default is 1500." }
                },
                "required": ["path"]
            }),
        },
        ToolDef {
            name: "writeFile",
            description: "Create a file or replace it when overwrite is true. Prefer editFile for partial changes. Do not submit more than one writeFile or editFile call for the same path in one turn.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "description": "Destination path." },
                    "content": { "type": "string", "description": "Complete file content." },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Allow a full replacement of an existing file."
                    }
                },
                "required": ["path", "content"]
            }),
        },
        ToolDef {
            name: "editFile",
            description: "Apply ordered replace, insert, or delete actions to a file. The file is written only when every action succeeds. Prefer oldText because it stays anchored to content. Line numbers are adjusted after earlier insertions and deletions, but cannot address a line rewritten earlier in the same call. Put line-based actions in top-to-bottom order. Do not submit more than one writeFile or editFile call for the same path in one turn.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "description": "File to update." },
                    "edits": {
                        "type": "array",
                        "description": "Actions apply sequentially. The file stays unchanged if any action fails.",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "action": { "type": "string", "enum": ["replace", "insert", "delete"], "description": "Operation to apply." },
                                "oldText": { "type": "string", "description": "Text to replace or delete." },
                                "newText": { "type": "string", "description": "Replacement or inserted text." },
                                "startLine": { "type": "number", "description": "First original line to change or use as an insert anchor." },
                                "endLine": { "type": "number", "description": "Last original line to change." },
                                "position": { "type": "string", "enum": ["before", "after"], "description": "Insert position." },
                                "anchorText": { "type": "string", "description": "Text in the line to use as an insert anchor." }
                            },
                            "required": ["action"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }),
        },
        ToolDef {
            name: "findFiles",
            description: "Search file names or file contents under a directory.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "description": "Directory to search." },
                    "pattern": { "type": "string", "description": "Regular expression or text to find." },
                    "mode": { "type": "string", "enum": ["name", "content"], "description": "Search names or contents." }
                },
                "required": ["pattern", "mode"]
            }),
        },
        ToolDef {
            name: "listFiles",
            description: "List files and directories under a path while respecting .gitignore rules.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string", "description": "Directory to list." },
                    "depth": { "type": "number", "description": "Directory depth to include. The default is 1." }
                },
                "required": ["path"]
            }),
        },
        ToolDef {
            name: "runShell",
            description: "Run a shell command in the foreground or background. Use a dedicated tool when one exists. Tool calls in the same turn can run concurrently, so do not submit commands that may modify the same files or processes.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "command": { "type": "string", "description": "Shell command to run." },
                    "workingDirectory": { "type": "string", "description": "Working directory." },
                    "env": { "type": "object", "description": "Additional environment variables." },
                    "timeout": { "type": "number", "description": "Timeout in milliseconds. The default is 300000." },
                    "background": {
                        "type": "boolean",
                        "description": "Return immediately with a job ID and log path. The agent receives an exit event when the process finishes."
                    }
                },
                "required": ["command"]
            }),
        },
        ToolDef {
            name: "fetchUrl",
            description: "Fetch an HTTP or HTTPS URL. By default, HTML is reduced to readable text.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "url": { "type": "string", "description": "URL to fetch." },
                    "rawContent": { "type": "boolean", "description": "Return the response without reducing HTML." },
                    "limit": { "type": "number", "description": "Maximum characters to return. The default is 20000." }
                },
                "required": ["url"]
            }),
        },
        ToolDef {
            name: "searchWeb",
            description: "Search through Tavily when TAVILY_API_KEY is set, or DuckDuckGo otherwise. Return result titles, snippets, and source URLs.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "query": { "type": "string", "description": "Search query." },
                    "depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Tavily search depth. The default is basic."
                    },
                    "maxResults": { "type": "number", "description": "Maximum results. The default is 5 and the limit is 20." },
                    "includeAnswer": {
                        "type": "boolean",
                        "description": "Ask Tavily to include its generated answer. The default is false."
                    },
                    "includeDomains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Only include these domains."
                    },
                    "excludeDomains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Exclude these domains."
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "manageTodos",
            description: "Add, list, complete, update, delete, or clear items in a JSON todo file.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "complete", "delete", "update", "clear"],
                        "description": "Operation to perform."
                    },
                    "text": {
                        "type": "string",
                        "description": "Todo text. Required for add."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Priority. The default is medium."
                    },
                    "dueDate": {
                        "type": "string",
                        "description": "Due date in ISO 8601 format."
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category."
                    },
                    "id": {
                        "type": "string",
                        "description": "Todo ID. Required for complete, delete, and update."
                    },
                    "completed": {
                        "type": "boolean",
                        "description": "Completion value for update."
                    },
                    "filter": {
                        "type": "string",
                        "enum": ["all", "pending", "completed"],
                        "description": "Filter used by list."
                    },
                    "sortBy": {
                        "type": "string",
                        "enum": ["createdAt", "priority", "dueDate"],
                        "description": "Sort field used by list."
                    },
                    "todoFile": {
                        "type": "string",
                        "description": "Todo file path. The default comes from the agent storage configuration."
                    }
                },
                "required": ["action"]
            }),
        },
        ToolDef {
            name: "loadSkill",
            description: "List, search, or load instruction sets for a task.",
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "load", "search"],
                        "description": "Operation to perform."
                    },
                    "argument": {
                        "type": "string",
                        "description": "Skill name for load, search query for search, or omitted for list."
                    }
                },
                "required": ["action"]
            }),
        },
    ]
}

/// Resolve a relative path within the sandbox root, rejecting absolute paths and directory traversal.
///
/// # Errors
///
/// Returns `ToolError::Denied` if the path is absolute or attempts to escape `root`.
/// Returns `ToolError::NotFound` if `root` cannot be resolved.
/// Returns `ToolError::Failed` if canonicalization fails for an existing path.
pub fn resolve_in(root: &Path, p: &str) -> Result<PathBuf, ToolError> {
    if p.is_empty() {
        let canonical_root = root.canonicalize().map_err(|e| ToolError::NotFound(e.to_string()))?;
        return Ok(canonical_root);
    }

    let req_path = Path::new(p);
    if req_path.is_absolute() {
        return Err(ToolError::Denied("absolute paths are not allowed".to_string()));
    }

    let canonical_root = root
        .canonicalize()
        .map_err(|e| ToolError::NotFound(format!("sandbox root not found: {e}")))?;

    let mut target = canonical_root.clone();
    for comp in req_path.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if target == canonical_root {
                    return Err(ToolError::Denied("path traversal out of sandbox".to_string()));
                }
                target.pop();
            }
            std::path::Component::Normal(c) => {
                target.push(c);
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(ToolError::Denied("absolute paths are not allowed".to_string()));
            }
        }
    }

    if !target.starts_with(&canonical_root) {
        return Err(ToolError::Denied("path escapes sandbox".to_string()));
    }

    if target.exists() {
        let canonical_target = target.canonicalize().map_err(|e| ToolError::Failed(e.to_string()))?;
        if !canonical_target.starts_with(&canonical_root) {
            return Err(ToolError::Denied("symlink escapes sandbox".to_string()));
        }
        Ok(canonical_target)
    } else {
        deny_dangling_link(&target, &canonical_root)?;
        let mut ancestor = target.as_path();
        while let Some(parent) = ancestor.parent() {
            if parent.exists() {
                let canonical_parent = parent.canonicalize().map_err(|e| ToolError::Failed(e.to_string()))?;
                if !canonical_parent.starts_with(&canonical_root) {
                    return Err(ToolError::Denied("symlink escapes sandbox".to_string()));
                }
                break;
            }
            ancestor = parent;
        }
        Ok(target)
    }
}

fn deny_dangling_link(target: &Path, canonical_root: &Path) -> Result<(), ToolError> {
    let Ok(meta) = std::fs::symlink_metadata(target) else {
        return Ok(());
    };
    if !meta.file_type().is_symlink() {
        return Ok(());
    }
    let link = std::fs::read_link(target).map_err(|err| ToolError::Failed(err.to_string()))?;
    let joined = if link.is_absolute() {
        link
    } else {
        target.parent().unwrap_or(canonical_root).join(link)
    };
    let mut ancestor = joined.as_path();
    loop {
        if ancestor.exists() {
            let canonical = ancestor
                .canonicalize()
                .map_err(|err| ToolError::Failed(err.to_string()))?;
            if !canonical.starts_with(canonical_root) {
                return Err(ToolError::Denied("symlink escapes sandbox".to_string()));
            }
            return Ok(());
        }
        match ancestor.parent() {
            Some(parent) => ancestor = parent,
            None => return Err(ToolError::Denied("symlink escapes sandbox".to_string())),
        }
    }
}

const MAX_READ_BYTES: u64 = 10 * 1024 * 1024;
const MAX_WRITE_BYTES: usize = 10 * 1024 * 1024;
const MAX_SEARCH_BYTES: u64 = 1024 * 1024;
const DEFAULT_PAGE_LINES: usize = 1500;

/// Dispatch a tool call by name. File tools are implemented here; the
/// remaining five return `Failed("unknown tool: ...")` until later tasks.
///
/// # Errors
///
/// Returns `ToolError::Failed` for unknown tool names or malformed inputs.
/// File tools can also return `Denied`, `NotFound`, or `TooLarge`.
pub async fn execute_tool(name: &str, input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    match name {
        "readFile" => exec_read_file(input, ctx).await,
        "writeFile" => exec_write_file(input, ctx).await,
        "editFile" => exec_edit_file(input, ctx).await,
        "findFiles" => exec_find_files(input, ctx),
        "listFiles" => exec_list_files(input, ctx),
        _ => Err(ToolError::Failed(format!("unknown tool: {name}"))),
    }
}

fn num_value(value: &serde_json::Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|n| u64::try_from(n).ok()))
}

fn to_usize(n: u64) -> usize {
    usize::try_from(n).unwrap_or(usize::MAX)
}

fn map_io_err(err: &std::io::Error, rel: &str) -> ToolError {
    match err.kind() {
        std::io::ErrorKind::NotFound => ToolError::NotFound(rel.to_string()),
        _ => ToolError::Failed(err.to_string()),
    }
}

fn media_mime_for_ext(ext: &str) -> Option<&'static str> {
    match ext {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "pdf" => Some("application/pdf"),
        "mp4" | "m4v" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        "mov" => Some("video/quicktime"),
        "avi" => Some("video/x-msvideo"),
        "mkv" => Some("video/x-matroska"),
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        "m4a" => Some("audio/mp4"),
        "flac" => Some("audio/flac"),
        "aac" => Some("audio/aac"),
        "ogg" | "oga" => Some("audio/ogg"),
        _ => None,
    }
}

fn sniff_media_mime(sample: &[u8]) -> Option<&'static str> {
    if sample.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some("image/png");
    }
    if sample.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    if sample.starts_with(b"GIF87a") || sample.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if sample.len() >= 12 && sample[..4] == *b"RIFF" && sample[8..12] == *b"WEBP" {
        return Some("image/webp");
    }
    if sample.starts_with(b"%PDF") {
        return Some("application/pdf");
    }
    if sample.len() >= 8 && sample[4..8] == *b"ftyp" {
        return Some("video/mp4");
    }
    if sample.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Some("video/webm");
    }
    if sample.starts_with(b"OggS") {
        return Some("audio/ogg");
    }
    if sample.len() >= 12 && sample[..4] == *b"RIFF" && sample[8..12] == *b"WAVE" {
        return Some("audio/wav");
    }
    if sample.starts_with(b"ID3") {
        return Some("audio/mpeg");
    }
    if sample.len() >= 2 && sample[0] == 0xFF && (sample[1] & 0xE0) == 0xE0 {
        return Some("audio/mpeg");
    }
    None
}

fn is_binary_sample(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return false;
    }
    let mut control = 0_usize;
    for &byte in sample {
        if byte == 0 {
            return true;
        }
        if byte < 9 || (14..=31).contains(&byte) {
            control += 1;
        }
    }
    control * 10 > sample.len() * 3
}

fn hex_preview(sample: &[u8]) -> String {
    sample
        .iter()
        .take(64)
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

async fn exec_read_file(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let rel = input
        .get("path")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("readFile: missing 'path'".to_string()))?;
    let target = resolve_in(&ctx.sandbox_root, rel)?;
    let meta = tokio::fs::metadata(&target)
        .await
        .map_err(|err| map_io_err(&err, rel))?;
    if !meta.is_file() {
        return Err(ToolError::Failed(format!("readFile: not a file: {rel}")));
    }
    if meta.len() > MAX_READ_BYTES {
        return Err(ToolError::TooLarge(format!("{rel}: {} bytes", meta.len())));
    }
    let bytes = tokio::fs::read(&target).await.map_err(|err| map_io_err(&err, rel))?;
    let ext = target
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or_default()
        .to_lowercase();
    let sample = &bytes[..bytes.len().min(4096)];
    if let Some(mime) = media_mime_for_ext(&ext).or_else(|| sniff_media_mime(sample)) {
        let encoded = base64::prelude::BASE64_STANDARD.encode(&bytes);
        return Ok(serde_json::json!({"mime_type": mime, "base64": encoded}).to_string());
    }
    if is_binary_sample(sample) {
        return Ok(format!(
            "{rel}: binary, {} bytes\nhex: {}",
            bytes.len(),
            hex_preview(sample)
        ));
    }
    let text = String::from_utf8_lossy(&bytes);
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines: Vec<&str> = normalized.split('\n').collect();
    if lines.last().is_some_and(|last| last.is_empty()) {
        lines.pop();
    }
    let offset = input.get("offset").and_then(num_value).map(to_usize);
    let start_line = input
        .get("startLine")
        .and_then(num_value)
        .map_or(0, |n| to_usize(n.saturating_sub(1)));
    let start = offset.unwrap_or(start_line);
    let mut limit = input
        .get("limit")
        .or_else(|| input.get("maxLines"))
        .and_then(num_value)
        .map_or(DEFAULT_PAGE_LINES, to_usize);
    if let Some(end) = input.get("endLine").and_then(num_value).map(to_usize) {
        limit = limit.min(end.saturating_sub(start));
    }
    let total = lines.len();
    let begin = start.min(total);
    let stop = start.saturating_add(limit).min(total);
    let numbered: Vec<String> = lines[begin..stop]
        .iter()
        .enumerate()
        .map(|(pos, line)| format!("{:>6}\t{line}", begin + pos + 1))
        .collect();
    let mut result = numbered.join("\n");
    if stop < total {
        result.push_str("\n[... truncated]");
    }
    Ok(result)
}

async fn exec_write_file(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let rel = input
        .get("path")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("writeFile: missing 'path'".to_string()))?;
    let content = input
        .get("content")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("writeFile: missing 'content'".to_string()))?;
    let allow = input
        .get("allow_overwrite")
        .or_else(|| input.get("overwrite"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let target = resolve_in(&ctx.sandbox_root, rel)?;
    if content.len() > MAX_WRITE_BYTES {
        return Err(ToolError::TooLarge(format!("{rel}: {} bytes", content.len())));
    }
    if tokio::fs::symlink_metadata(&target).await.is_ok() && !allow {
        return Err(ToolError::Denied(format!(
            "{rel} already exists (set allow_overwrite to overwrite)"
        )));
    }
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| ToolError::Failed(err.to_string()))?;
    }
    tokio::fs::write(&target, content)
        .await
        .map_err(|err| map_io_err(&err, rel))?;
    Ok(format!("Wrote {rel} ({} bytes).", content.len()))
}

fn old_text_of(op: &serde_json::Value) -> Option<&str> {
    op.get("oldText")
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.is_empty())
}

fn new_text_of(op: &serde_json::Value, index: usize) -> Result<&str, ToolError> {
    op.get("newText")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed(format!("edit[{index}]: replace/insert requires 'newText'")))
}

fn check_unique(content: &str, old: &str, index: usize) -> Result<(), ToolError> {
    let count = content.matches(old).count();
    if count == 0 {
        return Err(ToolError::Failed(format!("edit[{index}]: 'oldText' not found")));
    }
    if count > 1 {
        return Err(ToolError::Failed(format!(
            "edit[{index}]: 'oldText' matches {count} times"
        )));
    }
    Ok(())
}

fn edit_line_range(op: &serde_json::Value, index: usize, total: usize) -> Result<(usize, usize), ToolError> {
    let start = op.get("startLine").and_then(num_value).map(to_usize);
    let end = op.get("endLine").and_then(num_value).map(to_usize);
    match (start, end) {
        (Some(first), Some(last)) if first >= 1 && first <= last && last <= total => Ok((first, last)),
        (Some(_), Some(_)) => Err(ToolError::Failed(format!(
            "edit[{index}]: line range out of bounds (file has {total} lines)"
        ))),
        _ => Err(ToolError::Failed(format!(
            "edit[{index}]: requires 'oldText' or 'startLine'+'endLine'"
        ))),
    }
}

fn apply_replace(content: &mut String, op: &serde_json::Value, index: usize) -> Result<(), ToolError> {
    let new_text = new_text_of(op, index)?;
    if let Some(old) = old_text_of(op) {
        check_unique(content, old, index)?;
        *content = content.replacen(old, new_text, 1);
        return Ok(());
    }
    let total = content.split('\n').count();
    let (first, last) = edit_line_range(op, index, total)?;
    let mut lines: Vec<&str> = content.split('\n').collect();
    lines.splice(first - 1..last, new_text.split('\n'));
    let joined = lines.join("\n");
    *content = joined;
    Ok(())
}

fn apply_insert(content: &mut String, op: &serde_json::Value, index: usize) -> Result<(), ToolError> {
    let new_text = new_text_of(op, index)?;
    let position = op
        .get("position")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("after");
    if position != "before" && position != "after" {
        return Err(ToolError::Failed(format!(
            "edit[{index}]: unknown position '{position}'"
        )));
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    let at = if let Some(anchor) = op
        .get("anchorText")
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.is_empty())
    {
        let found = lines
            .iter()
            .position(|line| line.contains(anchor))
            .ok_or_else(|| ToolError::Failed(format!("edit[{index}]: 'anchorText' not found")))?;
        if position == "after" { found + 1 } else { found }
    } else if let Some(n) = op
        .get("line")
        .or_else(|| op.get("startLine"))
        .and_then(num_value)
        .map(to_usize)
    {
        if n < 1 || n > lines.len() {
            return Err(ToolError::Failed(format!(
                "edit[{index}]: line {n} out of range (file has {} lines)",
                lines.len()
            )));
        }
        if position == "after" { n } else { n - 1 }
    } else {
        return Err(ToolError::Failed(format!(
            "edit[{index}]: insert requires 'anchorText' or 'line'"
        )));
    };
    lines.splice(at..at, new_text.split('\n'));
    let joined = lines.join("\n");
    *content = joined;
    Ok(())
}

fn apply_delete(content: &mut String, op: &serde_json::Value, index: usize) -> Result<(), ToolError> {
    if let Some(old) = old_text_of(op) {
        check_unique(content, old, index)?;
        *content = content.replacen(old, "", 1);
        return Ok(());
    }
    let total = content.split('\n').count();
    let (first, last) = edit_line_range(op, index, total)?;
    let mut lines: Vec<&str> = content.split('\n').collect();
    lines.drain(first - 1..last);
    let joined = lines.join("\n");
    *content = joined;
    Ok(())
}

fn apply_edit_op(content: &mut String, op: &serde_json::Value, index: usize) -> Result<(), ToolError> {
    let action = op
        .get("op")
        .or_else(|| op.get("action"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed(format!("edit[{index}]: missing 'op'")))?;
    match action {
        "replace" => apply_replace(content, op, index),
        "insert" => apply_insert(content, op, index),
        "delete" => apply_delete(content, op, index),
        other => Err(ToolError::Failed(format!("edit[{index}]: unknown op '{other}'"))),
    }
}

async fn exec_edit_file(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let rel = input
        .get("path")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("editFile: missing 'path'".to_string()))?;
    let target = resolve_in(&ctx.sandbox_root, rel)?;
    let raw = tokio::fs::read(&target).await.map_err(|err| map_io_err(&err, rel))?;
    let text = String::from_utf8(raw).map_err(|_| ToolError::Failed(format!("editFile: not UTF-8 text: {rel}")))?;
    let ops = input
        .get("ops")
        .or_else(|| input.get("edits"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| ToolError::Failed("editFile: missing 'ops'".to_string()))?;
    if ops.is_empty() {
        return Err(ToolError::Failed("editFile: 'ops' must not be empty".to_string()));
    }
    let crlf = text.contains("\r\n");
    let mut content = text.replace("\r\n", "\n");
    for (index, op) in ops.iter().enumerate() {
        apply_edit_op(&mut content, op, index)?;
    }
    let output = if crlf { content.replace('\n', "\r\n") } else { content };
    let size = output.len();
    tokio::fs::write(&target, output.as_bytes())
        .await
        .map_err(|err| map_io_err(&err, rel))?;
    Ok(format!("Updated {rel} ({size} bytes)."))
}

fn exec_find_files(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let dir = input.get("path").and_then(serde_json::Value::as_str).unwrap_or(".");
    let pattern = input
        .get("pattern")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("findFiles: missing 'pattern'".to_string()))?;
    let mode = input.get("mode").and_then(serde_json::Value::as_str).unwrap_or("name");
    if mode != "name" && mode != "content" {
        return Err(ToolError::Failed(format!("findFiles: unknown mode '{mode}'")));
    }
    let matcher = regex::RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .map_err(|err| ToolError::Failed(format!("findFiles: invalid pattern: {err}")))?;
    let root = resolve_in(&ctx.sandbox_root, dir)?;
    if !root.exists() {
        return Err(ToolError::NotFound(dir.to_string()));
    }
    if !root.is_dir() {
        return Err(ToolError::Failed(format!("findFiles: not a directory: {dir}")));
    }
    let mut hits = Vec::new();
    for entry in ignore::WalkBuilder::new(&root).build() {
        let entry = entry.map_err(|err| ToolError::Failed(err.to_string()))?;
        let Some(kind) = entry.file_type() else { continue };
        if !kind.is_file() {
            continue;
        }
        let path = entry.path();
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if mode == "name" {
            let name = path.file_name().map(|n| n.to_string_lossy());
            if name.as_deref().is_some_and(|n| matcher.is_match(n)) {
                hits.push(rel);
            }
        } else {
            let Ok(bytes) = std::fs::read(path) else { continue };
            if bytes.len() > to_usize(MAX_SEARCH_BYTES) {
                continue;
            }
            if bytes[..bytes.len().min(512)].contains(&0) {
                continue;
            }
            let Ok(text) = String::from_utf8(bytes) else { continue };
            for (number, line) in text.split('\n').enumerate() {
                if matcher.is_match(line) {
                    let snippet: String = line.trim().chars().take(100).collect();
                    hits.push(format!("{rel}:{}: {snippet}", number + 1));
                }
            }
        }
    }
    hits.sort();
    if hits.is_empty() {
        Ok("No matches found.".to_string())
    } else {
        Ok(hits.join("\n"))
    }
}

fn exec_list_files(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let dir = input.get("path").and_then(serde_json::Value::as_str).unwrap_or(".");
    let depth = input.get("depth").and_then(num_value).map_or(1, to_usize).max(1);
    let root = resolve_in(&ctx.sandbox_root, dir)?;
    if !root.exists() {
        return Err(ToolError::NotFound(dir.to_string()));
    }
    if !root.is_dir() {
        return Err(ToolError::Failed(format!("listFiles: not a directory: {dir}")));
    }
    let mut names = Vec::new();
    for entry in ignore::WalkBuilder::new(&root).max_depth(Some(depth)).build() {
        let entry = entry.map_err(|err| ToolError::Failed(err.to_string()))?;
        if entry.depth() == 0 {
            continue;
        }
        let path = entry.path();
        let mut rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if entry.file_type().is_some_and(|kind| kind.is_dir()) {
            rel.push('/');
        }
        names.push(rel);
    }
    names.sort();
    if names.is_empty() {
        Ok("(Empty directory)".to_string())
    } else {
        Ok(names.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_write_then_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        execute_tool(
            "writeFile",
            &json!({"path":"a.txt","content":"hi","allow_overwrite":true}),
            &ctx,
        )
        .await
        .unwrap();
        let out = execute_tool("readFile", &json!({"path":"a.txt"}), &ctx).await.unwrap();
        assert!(out.contains("hi"));
    }

    #[tokio::test]
    async fn test_write_absolute_path_denied() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("writeFile", &json!({"path": "/abs.txt", "content": "hi"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)));
    }

    #[tokio::test]
    async fn test_write_traversal_denied() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("writeFile", &json!({"path": "../evil.txt", "content": "hi"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)));
    }

    #[tokio::test]
    async fn test_read_missing_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("readFile", &json!({"path": "missing.txt"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_write_overwrite_guard() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        execute_tool("writeFile", &json!({"path": "a.txt", "content": "v1"}), &ctx)
            .await
            .unwrap();
        let err = execute_tool("writeFile", &json!({"path": "a.txt", "content": "v2"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)));
        execute_tool(
            "writeFile",
            &json!({"path": "a.txt", "content": "v2", "allow_overwrite": true}),
            &ctx,
        )
        .await
        .unwrap();
        let out = execute_tool("readFile", &json!({"path": "a.txt"}), &ctx).await.unwrap();
        assert!(out.contains("v2"));
        assert!(!out.contains("v1"));
    }

    #[tokio::test]
    async fn test_edit_replace_insert_delete() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        execute_tool(
            "writeFile",
            &json!({"path": "f.txt", "content": "one\ntwo\nthree\n"}),
            &ctx,
        )
        .await
        .unwrap();
        let out = execute_tool(
            "editFile",
            &json!({"path": "f.txt", "ops": [
                {"op": "replace", "oldText": "two", "newText": "TWO"},
                {"op": "insert", "line": 1, "position": "after", "newText": "mid"},
                {"op": "delete", "startLine": 4, "endLine": 4}
            ]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(out.contains("Updated"));
        let back = execute_tool("readFile", &json!({"path": "f.txt"}), &ctx).await.unwrap();
        assert!(back.contains("TWO"));
        assert!(back.contains("mid"));
        assert!(!back.contains("three"));
    }

    #[tokio::test]
    async fn test_edit_unknown_op_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        execute_tool("writeFile", &json!({"path": "f.txt", "content": "one\n"}), &ctx)
            .await
            .unwrap();
        let err = execute_tool(
            "editFile",
            &json!({"path": "f.txt", "ops": [{"op": "frobnicate"}]}),
            &ctx,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("unknown op")));
    }

    #[tokio::test]
    async fn test_read_media_returns_json() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&[0; 32]);
        std::fs::write(dir.path().join("img.png"), &png).unwrap();
        let out = execute_tool("readFile", &json!({"path": "img.png"}), &ctx)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value["mime_type"], "image/png");
        let decoded = base64::prelude::BASE64_STANDARD
            .decode(value["base64"].as_str().unwrap())
            .unwrap();
        assert_eq!(decoded, png);
    }

    #[tokio::test]
    async fn test_find_files_name_and_content() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        execute_tool(
            "writeFile",
            &json!({"path": "alpha.txt", "content": "hello world"}),
            &ctx,
        )
        .await
        .unwrap();
        execute_tool("writeFile", &json!({"path": "beta.txt", "content": "other"}), &ctx)
            .await
            .unwrap();
        execute_tool(
            "writeFile",
            &json!({"path": "sub/gamma.txt", "content": "say hello"}),
            &ctx,
        )
        .await
        .unwrap();
        let names = execute_tool(
            "findFiles",
            &json!({"path": ".", "pattern": "alpha", "mode": "name"}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(names.contains("alpha.txt"));
        assert!(!names.contains("beta.txt"));
        let hits = execute_tool(
            "findFiles",
            &json!({"path": ".", "pattern": "hello", "mode": "content"}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(hits.contains("alpha.txt:1:"));
        assert!(hits.contains("gamma.txt:1:"));
        let none = execute_tool(
            "findFiles",
            &json!({"path": ".", "pattern": "zzz-no-match", "mode": "name"}),
            &ctx,
        )
        .await
        .unwrap();
        assert_eq!(none, "No matches found.");
    }

    #[tokio::test]
    async fn test_list_files_one_level() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        execute_tool("writeFile", &json!({"path": "a.txt", "content": "x"}), &ctx)
            .await
            .unwrap();
        execute_tool("writeFile", &json!({"path": "sub/b.txt", "content": "y"}), &ctx)
            .await
            .unwrap();
        let out = execute_tool("listFiles", &json!({"path": "."}), &ctx).await.unwrap();
        assert!(out.contains("a.txt"));
        assert!(out.contains("sub/"));
        assert!(!out.contains("b.txt"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_dangling_symlink_outside_denied() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        std::os::unix::fs::symlink("/no-such-polyport-target-xyz", dir.path().join("evil")).unwrap();
        let err = execute_tool("readFile", &json!({"path": "evil"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)));
    }

    #[tokio::test]
    async fn test_unknown_tool_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("teleport", &json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("unknown tool")));
        let err = execute_tool("runShell", &json!({"command": "echo hi"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("unknown tool")));
    }

    #[test]
    fn test_manifest_lists_exactly_the_ten_kept_tools() {
        let names: Vec<_> = tool_defs().iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            vec![
                "readFile",
                "writeFile",
                "editFile",
                "findFiles",
                "listFiles",
                "runShell",
                "fetchUrl",
                "searchWeb",
                "manageTodos",
                "loadSkill"
            ]
        );
    }

    #[test]
    fn test_tool_ctx_default_and_test_in() {
        let default_ctx = ToolCtx::default();
        assert_eq!(default_ctx.shell_timeout_secs, 120);
        assert_eq!(default_ctx.fetch_timeout_secs, 30);
        assert_eq!(default_ctx.fetch_max_bytes, 2 * 1024 * 1024);

        let dir = tempfile::tempdir().unwrap();
        let test_ctx = ToolCtx::test_in(&dir);
        assert_eq!(test_ctx.sandbox_root, dir.path());
        assert_eq!(test_ctx.shell_timeout_secs, 120);

        let test_ctx2 = test_in(&dir);
        assert_eq!(test_ctx2.sandbox_root, dir.path());
    }

    #[test]
    fn test_resolve_in_containment() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let resolved = resolve_in(root, "sub/file.txt").unwrap();
        assert!(resolved.starts_with(root.canonicalize().unwrap()));

        let err_abs = resolve_in(root, "/etc/passwd").unwrap_err();
        assert!(matches!(err_abs, ToolError::Denied(_)));

        let err_traversal = resolve_in(root, "../outside.txt").unwrap_err();
        assert!(matches!(err_traversal, ToolError::Denied(_)));

        let err_deep_traversal = resolve_in(root, "a/../../outside.txt").unwrap_err();
        assert!(matches!(err_deep_traversal, ToolError::Denied(_)));

        let resolved_cur = resolve_in(root, ".").unwrap();
        assert_eq!(resolved_cur, root.canonicalize().unwrap());
    }
}
