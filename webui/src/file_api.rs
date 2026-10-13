use std::collections::HashMap;
use std::path::{Path, PathBuf};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[derive(Clone, Debug)]
pub struct FileState {
    pub sandbox_root: PathBuf,
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for comp in path.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if matches!(
                    normalized.components().next_back(),
                    Some(std::path::Component::Normal(_))
                ) {
                    normalized.pop();
                } else if !normalized.as_os_str().is_empty() && normalized.is_absolute() {
                    // Root directory parent remains root.
                } else {
                    normalized.push(comp);
                }
            }
            _ => normalized.push(comp),
        }
    }
    normalized
}

fn resolve_root(root: &Path) -> PathBuf {
    if root.is_absolute() {
        normalize_path(root)
    } else if let Ok(cwd) = std::env::current_dir() {
        normalize_path(&cwd.join(root))
    } else {
        normalize_path(root)
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        axum::Json(serde_json::json!({
            "error": message,
        })),
    )
        .into_response()
}

#[allow(clippy::unused_async)]
pub async fn file_handler(State(state): State<FileState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let Some(path) = q.get("path") else {
        return error_response(StatusCode::BAD_REQUEST, "Missing path parameter");
    };
    if path.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "Missing path parameter");
    }

    let canonical_root = state.sandbox_root.canonicalize().ok();
    let lexical_root = resolve_root(&state.sandbox_root);

    let req_path = Path::new(path);
    let lexical_target = if req_path.is_absolute() {
        normalize_path(req_path)
    } else {
        normalize_path(&lexical_root.join(req_path))
    };

    let is_within_lexical = lexical_target.starts_with(&lexical_root)
        || canonical_root.as_ref().is_some_and(|cr| lexical_target.starts_with(cr));
    if !is_within_lexical {
        return error_response(StatusCode::FORBIDDEN, "Access denied");
    }

    let Ok(meta) = std::fs::metadata(&lexical_target) else {
        return error_response(StatusCode::NOT_FOUND, "File not found");
    };
    if !meta.is_file() {
        return error_response(StatusCode::NOT_FOUND, "File not found");
    }

    if let Ok(canonical_target) = lexical_target.canonicalize() {
        let allowed = canonical_target.starts_with(&lexical_root)
            || canonical_root
                .as_ref()
                .is_some_and(|cr| canonical_target.starts_with(cr));
        if !allowed {
            return error_response(StatusCode::FORBIDDEN, "Access denied");
        }
    }

    match std::fs::read(&lexical_target) {
        Ok(bytes) => {
            let content = String::from_utf8_lossy(&bytes).into_owned();
            let name = req_path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            (
                StatusCode::OK,
                axum::Json(serde_json::json!({
                    "name": name,
                    "path": path,
                    "content": content,
                })),
            )
                .into_response()
        }
        Err(err) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use axum::extract::{Query, State};
    use axum::http::StatusCode;

    use super::{FileState, file_handler};

    #[tokio::test]
    async fn test_file_api_blocks_traversal() {
        let state = FileState {
            sandbox_root: PathBuf::from("/tmp/polyport-plan-w1-root"),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "/etc/passwd".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "Access denied");
    }

    #[tokio::test]
    async fn test_file_api_blocks_relative_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "../../etc/passwd".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "Access denied");
    }

    #[tokio::test]
    async fn test_file_api_blocks_sibling_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "../sibling/file.txt".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "Access denied");
    }

    #[tokio::test]
    async fn test_file_api_missing_param() {
        let dir = tempfile::tempdir().unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(State(state), Query([].into_iter().collect())).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "Missing path parameter");
    }

    #[tokio::test]
    async fn test_file_api_empty_param() {
        let dir = tempfile::tempdir().unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), String::new())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "Missing path parameter");
    }

    #[tokio::test]
    async fn test_file_api_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "nonexistent.txt".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "File not found");
    }

    #[tokio::test]
    async fn test_file_api_directory_not_file() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "subdir".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "File not found");
    }

    #[tokio::test]
    async fn test_file_api_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "Hello, world!").unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "hello.txt".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["name"], "hello.txt");
        assert_eq!(val["path"], "hello.txt");
        assert_eq!(val["content"], "Hello, world!");
    }

    #[tokio::test]
    async fn test_file_api_nested_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("nested.txt"), "Nested content").unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "sub/nested.txt".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["name"], "nested.txt");
        assert_eq!(val["path"], "sub/nested.txt");
        assert_eq!(val["content"], "Nested content");
    }

    #[tokio::test]
    async fn test_file_api_binary_file_lossy() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = [
            b'h', b'e', b'l', b'l', b'o', b' ', 0xff, 0xfe, b' ', b'w', b'o', b'r', b'l', b'd',
        ];
        std::fs::write(dir.path().join("binary.bin"), bytes).unwrap();
        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "binary.bin".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["name"], "binary.bin");
        assert_eq!(val["path"], "binary.bin");
        let content = val["content"].as_str().unwrap();
        assert!(content.contains('\u{FFFD}'));
        assert!(content.starts_with("hello "));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_file_api_blocks_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let target = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(target.path(), "secret").unwrap();
        let link_path = dir.path().join("link_to_secret");
        std::os::unix::fs::symlink(target.path(), &link_path).unwrap();

        let state = FileState {
            sandbox_root: dir.path().to_path_buf(),
        };
        let res = file_handler(
            State(state),
            Query([("path".into(), "link_to_secret".into())].into_iter().collect()),
        )
        .await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(res.into_body(), 1024).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(val["error"], "Access denied");
    }
}
