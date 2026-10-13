// https://github.com/af-t/agent-sdk/tree/v4.0.0/src/tools

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

#[cfg(test)]
mod tests {
    use super::*;

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
