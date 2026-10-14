// https://github.com/af-t/agent-sdk/tree/v4.0.0/src/tools

use base64::Engine as _;
use std::fmt::Write as _;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

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
    /// Test hook: permit loopback destinations (`127/8`, `::1`) so unit tests can
    /// reach an in-process stub server. Defaults to false; every other blocked
    /// range stays denied when this is set.
    pub fetch_allow_loopback: bool,
}

impl Default for ToolCtx {
    fn default() -> Self {
        Self {
            sandbox_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            http: reqwest::Client::new(),
            shell_timeout_secs: 120,
            fetch_timeout_secs: 30,
            fetch_max_bytes: 2 * 1024 * 1024,
            fetch_allow_loopback: false,
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

/// Dispatch a tool call by name. All ten manifest tools are implemented:
/// file tools, `runShell`, `fetchUrl`, `searchWeb`, `manageTodos`, and
/// `loadSkill`.
///
/// # Errors
///
/// Returns `ToolError::Failed` for unknown tool names or malformed inputs.
/// File tools can also return `Denied`, `NotFound`, or `TooLarge`.
/// `fetchUrl` returns `Denied` for SSRF-blocked targets.
pub async fn execute_tool(name: &str, input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    match name {
        "readFile" => exec_read_file(input, ctx).await,
        "writeFile" => exec_write_file(input, ctx).await,
        "editFile" => exec_edit_file(input, ctx).await,
        "findFiles" => exec_find_files(input, ctx),
        "listFiles" => exec_list_files(input, ctx),
        "runShell" => exec_run_shell(input, ctx).await,
        "fetchUrl" => exec_fetch_url(input, ctx).await,
        "searchWeb" => exec_search_web(input, ctx).await,
        "manageTodos" => exec_manage_todos(input, ctx).await,
        "loadSkill" => exec_load_skill(input),
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

const MAX_SHELL_OUTPUT_BYTES: usize = 32 * 1024;

// Env keep-list: no *_TOKEN, *_KEY, or *_SECRET name is exempt; PATH and other harmless vars pass through.
const SHELL_ENV_KEEP: &[&str] = &[];

fn is_secret_env_key(key: &str) -> bool {
    if SHELL_ENV_KEEP.contains(&key) {
        return false;
    }
    let upper = key.to_ascii_uppercase();
    upper.ends_with("_TOKEN") || upper.ends_with("_KEY") || upper.ends_with("_SECRET")
}

async fn read_pipe<R>(pipe: Option<R>) -> Vec<u8>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = tokio::io::AsyncReadExt::read_to_end(&mut pipe, &mut buf).await;
    }
    buf
}

#[cfg(windows)]
fn build_shell(cmd: &str) -> tokio::process::Command {
    let mut command = tokio::process::Command::new("cmd");
    command.arg("/C").arg(cmd);
    command
}

#[cfg(not(windows))]
fn build_shell(cmd: &str) -> tokio::process::Command {
    let mut command = tokio::process::Command::new("sh");
    command.arg("-c").arg(cmd);
    command.process_group(0);
    command
}

/// Run `cmd` in the foreground shell with cwd jailed to the sandbox root.
///
/// Stdout and stderr are merged and truncated to 32KB. Overrunning `timeout_secs`
/// kills the child and returns `Failed` containing "timed out".
///
/// # Errors
///
/// Returns `ToolError::Failed` for empty commands, spawn failures, or timeouts.
pub async fn run_shell_checked(ctx: &ToolCtx, cmd: &str, timeout_secs: u64) -> Result<String, ToolError> {
    if cmd.trim().is_empty() {
        return Err(ToolError::Failed("runShell: empty command".to_string()));
    }
    tokio::fs::create_dir_all(&ctx.sandbox_root)
        .await
        .map_err(|err| ToolError::Failed(format!("runShell: sandbox root unavailable: {err}")))?;
    let mut command = build_shell(cmd);
    command
        .current_dir(&ctx.sandbox_root)
        .env_clear()
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    for (key, value) in std::env::vars() {
        if !is_secret_env_key(&key) {
            command.env(&key, &value);
        }
    }
    let mut child = command.spawn().map_err(|err| ToolError::Failed(err.to_string()))?;
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let run = async {
        let stdout_task = tokio::spawn(read_pipe(stdout_pipe));
        let stderr_task = tokio::spawn(read_pipe(stderr_pipe));
        child.wait().await.map_err(|err| ToolError::Failed(err.to_string()))?;
        let stdout = stdout_task.await.map_err(|err| ToolError::Failed(err.to_string()))?;
        let stderr = stderr_task.await.map_err(|err| ToolError::Failed(err.to_string()))?;
        Ok::<(Vec<u8>, Vec<u8>), ToolError>((stdout, stderr))
    };
    let (stdout, stderr) = match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), run).await {
        Ok(Ok((stdout, stderr))) => (stdout, stderr),
        Ok(Err(err)) => return Err(err),
        Err(_) => {
            #[cfg(unix)]
            if let Some(pid) = child.id()
                && let Ok(pid_i32) = i32::try_from(pid)
            {
                unsafe {
                    libc::kill(-pid_i32, libc::SIGKILL);
                }
            }
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(ToolError::Failed(format!("runShell: timed out after {timeout_secs}s")));
        }
    };
    let mut merged = String::from_utf8_lossy(&stdout).into_owned();
    merged.push_str(&String::from_utf8_lossy(&stderr));
    if merged.len() > MAX_SHELL_OUTPUT_BYTES {
        let mut end = MAX_SHELL_OUTPUT_BYTES;
        while !merged.is_char_boundary(end) {
            end -= 1;
        }
        merged.truncate(end);
    }
    Ok(merged)
}

async fn exec_run_shell(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let cmd = input
        .get("command")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("runShell: missing 'command'".to_string()))?;
    let timeout_secs = input
        .get("timeout_secs")
        .and_then(num_value)
        .or_else(|| input.get("timeout").and_then(num_value).map(|ms| ms.div_ceil(1000)))
        .unwrap_or(ctx.shell_timeout_secs);
    run_shell_checked(ctx, cmd, timeout_secs).await
}

const MAX_FETCH_REDIRECTS: usize = 3;
const DEFAULT_SEARCH_RESULTS: usize = 5;
const MAX_SEARCH_RESULTS: usize = 20;

fn is_blocked_ipv4(octets: [u8; 4], allow_loopback: bool) -> bool {
    let a = octets[0];
    let b = octets[1];
    let c = octets[2];
    if a == 127 {
        return !allow_loopback;
    }
    if a == 0 || a == 10 {
        return true;
    }
    if a == 172 && (16..=31).contains(&b) {
        return true;
    }
    if a == 192 && b == 168 {
        return true;
    }
    if a == 169 && b == 254 {
        return true;
    }
    if a == 100 && (64..=127).contains(&b) {
        return true;
    }
    if a == 192 && b == 0 && (c == 0 || c == 2) {
        return true;
    }
    if a == 198 && (b == 18 || b == 19) {
        return true;
    }
    if a == 198 && b == 51 && c == 100 {
        return true;
    }
    if a == 203 && b == 0 && c == 113 {
        return true;
    }
    if a >= 224 {
        return true;
    }
    false
}

fn is_blocked_ipv6(segments: [u16; 8], allow_loopback: bool) -> bool {
    if segments == [0, 0, 0, 0, 0, 0, 0, 0] {
        return true;
    }
    if segments == [0, 0, 0, 0, 0, 0, 0, 1] {
        return !allow_loopback;
    }
    let top = segments[0];
    if top & 0xffc0 == 0xfe80 {
        return true;
    }
    if top & 0xfe00 == 0xfc00 {
        return true;
    }
    if top & 0xff00 == 0xff00 {
        return true;
    }
    if segments[0] == 0x2001 && segments[1] == 0x0db8 {
        return true;
    }
    false
}

fn is_blocked_ip(ip: &IpAddr, allow_loopback: bool) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_ipv4(v4.octets(), allow_loopback),
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_blocked_ipv4(mapped.octets(), allow_loopback);
            }
            is_blocked_ipv6(v6.segments(), allow_loopback)
        }
    }
}

async fn check_url_allowed(url: &reqwest::Url, ctx: &ToolCtx) -> Result<(), ToolError> {
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(ToolError::Denied(format!(
            "fetchUrl: scheme '{}' not allowed",
            url.scheme()
        )));
    }
    let host = url.host_str().unwrap_or_default();
    if host.is_empty() {
        return Err(ToolError::Denied("fetchUrl: URL has no host".to_string()));
    }
    if host.eq_ignore_ascii_case("localhost") {
        return Err(ToolError::Denied("fetchUrl: localhost is not allowed".to_string()));
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_blocked_ip(&ip, ctx.fetch_allow_loopback) {
            return Err(ToolError::Denied(format!("fetchUrl: blocked IP {ip}")));
        }
        return Ok(());
    }
    if host.contains(':') {
        return Err(ToolError::Denied(format!("fetchUrl: blocked IP literal {host}")));
    }
    let port = url.port_or_known_default().unwrap_or(80);
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|err| ToolError::Denied(format!("fetchUrl: cannot resolve {host}: {err}")))?;
    let mut resolved = false;
    for addr in addrs {
        resolved = true;
        if is_blocked_ip(&addr.ip(), ctx.fetch_allow_loopback) {
            return Err(ToolError::Denied(format!(
                "fetchUrl: {host} resolves to blocked IP {}",
                addr.ip()
            )));
        }
    }
    if !resolved {
        return Err(ToolError::Denied(format!("fetchUrl: {host} resolved to no addresses")));
    }
    Ok(())
}

fn is_redirect(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 301 | 302 | 303 | 307 | 308)
}

async fn read_body_capped(resp: reqwest::Response, cap: usize) -> Result<(Vec<u8>, bool), ToolError> {
    let mut resp = resp;
    let mut buf = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = resp.chunk().await.map_err(|err| ToolError::Failed(err.to_string()))? {
        let room = cap.saturating_sub(buf.len());
        if room == 0 {
            truncated = true;
            break;
        }
        if chunk.len() <= room {
            buf.extend_from_slice(&chunk);
        } else {
            buf.extend_from_slice(&chunk[..room]);
            truncated = true;
        }
    }
    Ok((buf, truncated))
}

fn capped_text(buf: &[u8], truncated: bool) -> String {
    let mut text = String::from_utf8_lossy(buf).into_owned();
    if truncated && text.ends_with('\u{FFFD}') {
        text.pop();
    }
    text
}

async fn fetch_bytes_capped(start: &str, ctx: &ToolCtx) -> Result<(reqwest::StatusCode, Vec<u8>, bool), ToolError> {
    let mut current =
        reqwest::Url::parse(start).map_err(|err| ToolError::Failed(format!("fetchUrl: invalid URL: {err}")))?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|err| ToolError::Failed(err.to_string()))?;
    let mut redirects = 0_usize;
    loop {
        check_url_allowed(&current, ctx).await?;
        let resp = client
            .get(current.clone())
            .send()
            .await
            .map_err(|err| ToolError::Failed(err.to_string()))?;
        if is_redirect(resp.status()) {
            redirects += 1;
            if redirects > MAX_FETCH_REDIRECTS {
                return Err(ToolError::Failed(format!(
                    "fetchUrl: too many redirects (max {MAX_FETCH_REDIRECTS})"
                )));
            }
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            if location.is_empty() {
                return Err(ToolError::Failed("fetchUrl: redirect without location".to_string()));
            }
            current = current
                .join(&location)
                .map_err(|err| ToolError::Failed(format!("fetchUrl: bad redirect target: {err}")))?;
            let _ = current.set_username("");
            let _ = current.set_password(None);
            continue;
        }
        let status = resp.status();
        let (body, truncated) = read_body_capped(resp, ctx.fetch_max_bytes).await?;
        return Ok((status, body, truncated));
    }
}

async fn exec_fetch_url(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let raw = input
        .get("url")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("fetchUrl: missing 'url'".to_string()))?;
    if raw.trim().is_empty() {
        return Err(ToolError::Failed("fetchUrl: empty 'url'".to_string()));
    }
    let timeout = Duration::from_secs(ctx.fetch_timeout_secs);
    let (status, body, truncated) = match tokio::time::timeout(timeout, fetch_bytes_capped(raw, ctx)).await {
        Ok(result) => result?,
        Err(_) => {
            return Err(ToolError::Failed(format!(
                "fetchUrl: timed out after {}s",
                ctx.fetch_timeout_secs
            )));
        }
    };
    if !status.is_success() {
        return Err(ToolError::Failed(format!(
            "fetchUrl: unexpected status {}",
            status.as_u16()
        )));
    }
    Ok(capped_text(&body, truncated))
}

fn collect_ddg_topics<'a>(node: &'a serde_json::Value, out: &mut Vec<(&'a str, &'a str)>) {
    if let Some(nested) = node.get("Topics").and_then(serde_json::Value::as_array) {
        for child in nested {
            collect_ddg_topics(child, out);
        }
        return;
    }
    let text = node.get("Text").and_then(serde_json::Value::as_str).unwrap_or_default();
    let url = node
        .get("FirstURL")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    out.push((text, url));
}

fn format_ddg_results(data: &serde_json::Value, max_results: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    let abstract_text = data
        .get("AbstractText")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim();
    if !abstract_text.is_empty() {
        lines.push(abstract_text.to_string());
    }
    if let Some(topics) = data.get("RelatedTopics").and_then(serde_json::Value::as_array) {
        let mut flat: Vec<(&str, &str)> = Vec::new();
        for topic in topics {
            collect_ddg_topics(topic, &mut flat);
        }
        for (text, url) in flat {
            if lines.len() >= max_results {
                break;
            }
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let url = url.trim();
            if url.is_empty() {
                lines.push(text.to_string());
            } else {
                lines.push(format!("{text} ({url})"));
            }
        }
    }
    if lines.is_empty() {
        return "no results".to_string();
    }
    lines.join("\n")
}

async fn exec_search_web(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let query = input
        .get("query")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("searchWeb: missing 'query'".to_string()))?;
    if query.trim().is_empty() {
        return Err(ToolError::Failed("searchWeb: empty 'query'".to_string()));
    }
    let max_results = input
        .get("maxResults")
        .and_then(num_value)
        .map_or(DEFAULT_SEARCH_RESULTS, to_usize)
        .min(MAX_SEARCH_RESULTS);
    let mut endpoint =
        reqwest::Url::parse("https://api.duckduckgo.com/").map_err(|err| ToolError::Failed(err.to_string()))?;
    endpoint
        .query_pairs_mut()
        .append_pair("q", query)
        .append_pair("format", "json");
    let timeout = Duration::from_secs(ctx.fetch_timeout_secs);
    let (status, body, _) = match tokio::time::timeout(timeout, fetch_bytes_capped(endpoint.as_str(), ctx)).await {
        Ok(result) => result?,
        Err(_) => {
            return Err(ToolError::Failed(format!(
                "searchWeb: timed out after {}s",
                ctx.fetch_timeout_secs
            )));
        }
    };
    if !status.is_success() {
        return Err(ToolError::Failed(format!(
            "searchWeb: DuckDuckGo failed ({})",
            status.as_u16()
        )));
    }
    let data: serde_json::Value =
        serde_json::from_slice(&body).map_err(|err| ToolError::Failed(format!("searchWeb: bad JSON: {err}")))?;
    Ok(format_ddg_results(&data, max_results))
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TodoItem {
    id: String,
    #[serde(default)]
    text: String,
    #[serde(default = "default_todo_priority")]
    priority: String,
    #[serde(default = "default_todo_category")]
    category: String,
    #[serde(default)]
    completed: bool,
    #[serde(rename = "createdAt", default)]
    created_at: String,
    #[serde(rename = "updatedAt", default)]
    updated_at: String,
    #[serde(rename = "dueDate", default)]
    due_date: Option<String>,
}

fn default_todo_priority() -> String {
    "medium".to_string()
}

fn default_todo_category() -> String {
    "general".to_string()
}

const MAX_TODO_ITEMS: usize = 1000;

fn priority_rank(priority: &str) -> u8 {
    match priority {
        "high" => 3,
        "medium" => 2,
        "low" => 1,
        _ => 0,
    }
}

fn days_to_civil(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = if month_part < 10 {
        month_part + 3
    } else {
        month_part - 9
    };
    (if month <= 2 { year + 1 } else { year }, month, day)
}

fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let day_secs = i64::try_from(secs).unwrap_or(0);
    let (year, month, day) = days_to_civil(day_secs.div_euclid(86_400));
    let rest = day_secs.rem_euclid(86_400);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        (rest % 3600) / 60,
        rest % 60
    )
}

fn generate_todo_id(todos: &[TodoItem]) -> String {
    const ALPHABET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.subsec_nanos());
    let salt = u32::try_from(todos.len()).unwrap_or(u32::MAX);
    let mut value = nanos ^ salt.wrapping_mul(0x9E37_79B9);
    let mut id = String::with_capacity(5);
    for _ in 0..5 {
        let index = usize::try_from(value % 36).unwrap_or(0);
        id.push(char::from(ALPHABET[index]));
        value /= 36;
    }
    let mut candidate = id.clone();
    let mut bump = 0_u32;
    while todos.iter().any(|todo| todo.id == candidate) {
        bump += 1;
        candidate = format!("{id}-{bump}");
    }
    candidate
}

async fn todos_file_path(input: &serde_json::Value, ctx: &ToolCtx) -> Result<PathBuf, ToolError> {
    let rel = input
        .get("todoFile")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("todos.json");
    if rel.trim().is_empty() {
        return Err(ToolError::Failed("manageTodos: empty 'todoFile'".to_string()));
    }
    tokio::fs::create_dir_all(&ctx.sandbox_root)
        .await
        .map_err(|err| ToolError::Failed(format!("manageTodos: sandbox root unavailable: {err}")))?;
    resolve_in(&ctx.sandbox_root, rel)
}

async fn read_todos(path: &Path) -> Result<Vec<TodoItem>, ToolError> {
    let raw = match tokio::fs::read_to_string(path).await {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(ToolError::Failed(format!("manageTodos: cannot read todo file: {err}"))),
    };
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| ToolError::Failed(format!("manageTodos: corrupt todo file: {err}")))?;
    let items = value
        .as_array()
        .ok_or_else(|| ToolError::Failed("manageTodos: todo file must contain a JSON array".to_string()))?;
    for item in items {
        if let Some(object) = item.as_object() {
            for legacy in ["created_at", "updated_at", "due_date"] {
                if object.contains_key(legacy) {
                    return Err(ToolError::Failed(
                        "manageTodos: todo file uses removed fields; rename created_at to createdAt, updated_at to updatedAt, and due_date to dueDate".to_string(),
                    ));
                }
            }
        }
    }
    serde_json::from_value::<Vec<TodoItem>>(value)
        .map_err(|err| ToolError::Failed(format!("manageTodos: corrupt todo file: {err}")))
}

async fn write_todos(path: &Path, todos: &[TodoItem]) -> Result<(), ToolError> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| ToolError::Failed(format!("manageTodos: cannot create todo dir: {err}")))?;
    }
    let pretty = serde_json::to_string_pretty(todos)
        .map_err(|err| ToolError::Failed(format!("manageTodos: cannot encode todos: {err}")))?;
    tokio::fs::write(path, pretty)
        .await
        .map_err(|err| ToolError::Failed(format!("manageTodos: cannot write todo file: {err}")))?;
    Ok(())
}

fn todo_priority_label(priority: &str) -> &'static str {
    match priority {
        "high" => "[high]",
        "low" => "[low]",
        _ => "[medium]",
    }
}

async fn todo_add(input: &serde_json::Value, path: &Path, todos: &mut Vec<TodoItem>) -> Result<String, ToolError> {
    let text = input
        .get("text")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("manageTodos: 'text' is required to add a todo".to_string()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(ToolError::Failed(
            "manageTodos: 'text' is required to add a todo".to_string(),
        ));
    }
    if todos.len() >= MAX_TODO_ITEMS {
        return Err(ToolError::Failed(format!(
            "manageTodos: maximum todo limit reached ({MAX_TODO_ITEMS})"
        )));
    }
    let priority = input
        .get("priority")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("medium");
    let category = input
        .get("category")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("general");
    let due_date = input
        .get("dueDate")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let now = now_iso();
    let item = TodoItem {
        id: generate_todo_id(todos),
        text: trimmed.to_string(),
        priority: priority.to_string(),
        category: category.to_string(),
        completed: false,
        created_at: now.clone(),
        updated_at: now,
        due_date,
    };
    let due_info = item
        .due_date
        .as_deref()
        .map_or(String::new(), |due| format!(" (Due: {due})"));
    let out = format!(
        "Todo added:\n   Text: {}\n   Priority: {}\n   Category: {}{due_info}\n   ID: {}",
        item.text,
        item.priority.to_uppercase(),
        item.category,
        item.id
    );
    todos.push(item);
    write_todos(path, todos).await?;
    Ok(out)
}

fn todo_list(input: &serde_json::Value, todos: &[TodoItem]) -> String {
    let filter = input.get("filter").and_then(serde_json::Value::as_str).unwrap_or("all");
    let sort_by = input
        .get("sortBy")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("createdAt");
    let mut view: Vec<&TodoItem> = todos.iter().collect();
    if filter == "pending" {
        view.retain(|todo| !todo.completed);
    } else if filter == "completed" {
        view.retain(|todo| todo.completed);
    }
    match sort_by {
        "priority" => view.sort_by(|a, b| {
            priority_rank(&b.priority)
                .cmp(&priority_rank(&a.priority))
                .then_with(|| b.created_at.cmp(&a.created_at))
        }),
        "dueDate" => view.sort_by(|a, b| match (&a.due_date, &b.due_date) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            (Some(first), Some(second)) => first.cmp(second),
        }),
        _ => view.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
    }
    let label = if filter == "all" {
        String::new()
    } else {
        format!(" ({filter})")
    };
    if view.is_empty() {
        return format!("No tasks{label}.");
    }
    let mut out = format!("Task List{label} - Total: {}\n", view.len());
    out.push_str(&"-".repeat(60));
    out.push('\n');
    for (index, todo) in view.iter().enumerate() {
        let status = if todo.completed { "[done]" } else { "[pending]" };
        let due_info = todo
            .due_date
            .as_deref()
            .map_or(String::new(), |due| format!(" | due: {due}"));
        let _ = write!(
            out,
            "{}. {status} {} {}\n   ID: {} | {} | Created: {}{due_info}\n",
            index + 1,
            todo_priority_label(&todo.priority),
            todo.text,
            todo.id,
            todo.category.to_uppercase(),
            todo.created_at,
        );
    }
    let total = todos.len();
    let completed = todos.iter().filter(|todo| todo.completed).count();
    let pending = total - completed;
    out.push_str(&"-".repeat(60));
    let _ = write!(out, "\nSummary: {completed}/{total} completed | {pending} pending");
    out
}

async fn todo_complete(input: &serde_json::Value, path: &Path, todos: &mut [TodoItem]) -> Result<String, ToolError> {
    let id = input
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("manageTodos: 'id' is required to complete a todo".to_string()))?;
    let text = {
        let todo = todos
            .iter_mut()
            .find(|todo| todo.id == id)
            .ok_or_else(|| ToolError::Failed(format!("manageTodos: todo with ID \"{id}\" not found")))?;
        if todo.completed {
            return Ok(format!("Todo \"{}\" is already completed.", todo.text));
        }
        todo.completed = true;
        todo.updated_at = now_iso();
        todo.text.clone()
    };
    write_todos(path, todos).await?;
    Ok(format!("Todo completed:\n   \"{text}\""))
}

async fn todo_delete(input: &serde_json::Value, path: &Path, todos: &mut Vec<TodoItem>) -> Result<String, ToolError> {
    let id = input
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("manageTodos: 'id' is required to delete a todo".to_string()))?;
    let index = todos
        .iter()
        .position(|todo| todo.id == id)
        .ok_or_else(|| ToolError::Failed(format!("manageTodos: todo with ID \"{id}\" not found")))?;
    let removed = todos.remove(index);
    write_todos(path, todos).await?;
    Ok(format!("Todo deleted:\n   \"{}\"", removed.text))
}

async fn todo_update(input: &serde_json::Value, path: &Path, todos: &mut [TodoItem]) -> Result<String, ToolError> {
    let id = input
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("manageTodos: 'id' is required to update a todo".to_string()))?;
    let snapshot = {
        let todo = todos
            .iter_mut()
            .find(|todo| todo.id == id)
            .ok_or_else(|| ToolError::Failed(format!("manageTodos: todo with ID \"{id}\" not found")))?;
        let mut changes: Vec<&str> = Vec::new();
        if let Some(text) = input.get("text").and_then(serde_json::Value::as_str) {
            todo.text = text.trim().to_string();
            changes.push("text");
        }
        if let Some(priority) = input.get("priority").and_then(serde_json::Value::as_str) {
            todo.priority = priority.to_string();
            changes.push("priority");
        }
        if let Some(category) = input.get("category").and_then(serde_json::Value::as_str) {
            todo.category = category.to_string();
            changes.push("category");
        }
        if input.as_object().is_some_and(|fields| fields.contains_key("dueDate")) {
            todo.due_date = input
                .get("dueDate")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            changes.push("due date");
        }
        if let Some(completed) = input.get("completed").and_then(serde_json::Value::as_bool) {
            todo.completed = completed;
            changes.push("status");
        }
        todo.updated_at = now_iso();
        (todo.clone(), changes.join(", "))
    };
    let (updated, changed) = snapshot;
    write_todos(path, todos).await?;
    if changed.is_empty() {
        return Ok(format!("No changes applied to todo \"{}\".", updated.text));
    }
    let status = if updated.completed {
        "[done] Completed"
    } else {
        "[pending] Pending"
    };
    let due_info = updated
        .due_date
        .as_deref()
        .map_or(String::new(), |due| format!(" | due: {due}"));
    Ok(format!(
        "Todo updated:\n   \"{}\"\n   Status: {status}\n   Priority: {} {}\n   Category: {}{due_info}\n   Changed: {changed}",
        updated.text,
        todo_priority_label(&updated.priority),
        updated.priority.to_uppercase(),
        updated.category.to_uppercase(),
    ))
}

async fn todo_clear(path: &Path, todos: &mut Vec<TodoItem>) -> Result<String, ToolError> {
    if todos.is_empty() {
        return Ok("Todo list is already empty.".to_string());
    }
    let count = todos.len();
    todos.clear();
    write_todos(path, todos).await?;
    Ok(format!("Cleared {count} todos."))
}

async fn exec_manage_todos(input: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let op = input
        .get("action")
        .or_else(|| input.get("op"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ToolError::Failed("manageTodos: missing 'action'".to_string()))?;
    let path = todos_file_path(input, ctx).await?;
    let mut todos = read_todos(&path).await?;
    match op {
        "add" => todo_add(input, &path, &mut todos).await,
        "list" => Ok(todo_list(input, &todos)),
        "complete" => todo_complete(input, &path, &mut todos).await,
        "delete" => todo_delete(input, &path, &mut todos).await,
        "update" => todo_update(input, &path, &mut todos).await,
        "clear" => todo_clear(&path, &mut todos).await,
        other => Err(ToolError::Failed(format!("manageTodos: unknown op '{other}'"))),
    }
}

fn list_skills() -> String {
    let names = crate::skills::skill_names();
    let mut out = format!("# Available Skills ({})\n", names.len());
    for name in names {
        let _ = write!(out, "\n- {name}");
    }
    out
}

fn load_one_skill(name: &str) -> Result<String, ToolError> {
    crate::skills::load_skill(name)
        .map(|body| format!("# {name}\n\n{body}"))
        .ok_or_else(|| ToolError::Failed(format!("loadSkill: skill \"{name}\" not found")))
}

fn search_skills(query: &str) -> String {
    let lowered = query.to_lowercase();
    let hits: Vec<&str> = crate::skills::skill_names()
        .into_iter()
        .filter(|name| {
            name.to_lowercase().contains(&lowered)
                || crate::skills::load_skill(name).is_some_and(|body| body.to_lowercase().contains(&lowered))
        })
        .collect();
    if hits.is_empty() {
        return format!("No skills found matching \"{query}\".\n\n{}", list_skills());
    }
    let mut out = format!("# Skills matching \"{query}\" ({})\n", hits.len());
    for name in hits {
        let _ = write!(out, "\n- {name}");
    }
    out
}

fn exec_load_skill(input: &serde_json::Value) -> Result<String, ToolError> {
    let action = input
        .get("action")
        .or_else(|| input.get("op"))
        .and_then(serde_json::Value::as_str);
    let argument = input
        .get("argument")
        .or_else(|| input.get("name"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim);
    match action {
        Some("list") => Ok(list_skills()),
        Some("load") => {
            let name = argument
                .filter(|arg| !arg.is_empty())
                .ok_or_else(|| ToolError::Failed("loadSkill: 'argument' is required to load a skill".to_string()))?;
            load_one_skill(name)
        }
        Some("search") => {
            let query = argument
                .filter(|arg| !arg.is_empty())
                .ok_or_else(|| ToolError::Failed("loadSkill: 'argument' is required to search skills".to_string()))?;
            Ok(search_skills(query))
        }
        Some(other) => Err(ToolError::Failed(format!("loadSkill: unknown action '{other}'"))),
        None => match argument {
            None | Some("") => Ok(list_skills()),
            Some(name) => load_one_skill(name),
        },
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
    async fn test_shell_echo_and_timeout() {
        let ctx = ToolCtx::test_in(&tempfile::tempdir().unwrap());
        let out = execute_tool("runShell", &json!({"command":"echo hello"}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("hello"));
    }

    #[tokio::test]
    async fn test_todos_add_and_list() {
        let ctx = ToolCtx::test_in(&tempfile::tempdir().unwrap());
        execute_tool("manageTodos", &json!({"op":"add","text":"write code"}), &ctx)
            .await
            .unwrap();
        let out = execute_tool("manageTodos", &json!({"op":"list"}), &ctx).await.unwrap();
        assert!(out.contains("write code"));
    }

    #[tokio::test]
    async fn test_unknown_tool_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("teleport", &json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("unknown tool")));
    }

    #[tokio::test]
    async fn test_todos_update_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let added = execute_tool("manageTodos", &json!({"action": "add", "text": "first task"}), &ctx)
            .await
            .unwrap();
        let id = added.rsplit("ID:").next().unwrap().trim().to_string();
        let updated = execute_tool(
            "manageTodos",
            &json!({"action": "update", "id": id, "text": "second task"}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(updated.contains("second task"));
        let out = execute_tool("manageTodos", &json!({"action": "list"}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("second task"));
        assert!(!out.contains("first task"));
    }

    #[tokio::test]
    async fn test_todos_complete_delete_clear() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let added = execute_tool("manageTodos", &json!({"op": "add", "text": "finish me"}), &ctx)
            .await
            .unwrap();
        let id = added.rsplit("ID:").next().unwrap().trim().to_string();
        let done = execute_tool("manageTodos", &json!({"op": "complete", "id": id}), &ctx)
            .await
            .unwrap();
        assert!(done.contains("finish me"));
        let pending = execute_tool("manageTodos", &json!({"op": "list", "filter": "pending"}), &ctx)
            .await
            .unwrap();
        assert!(pending.contains("No tasks"));
        let cleared = execute_tool("manageTodos", &json!({"op": "clear"}), &ctx)
            .await
            .unwrap();
        assert!(cleared.contains("Cleared 1 todos"));
        let deleted_ctx_out = execute_tool("manageTodos", &json!({"op": "list"}), &ctx).await.unwrap();
        assert!(deleted_ctx_out.contains("No tasks"));
        let missing = execute_tool("manageTodos", &json!({"op": "delete", "id": id}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(missing, ToolError::Failed(ref msg) if msg.contains("not found")));
    }

    #[tokio::test]
    async fn test_todos_unknown_op_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("manageTodos", &json!({"op": "frobnicate"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("unknown op")));
        let err = execute_tool("manageTodos", &json!({"action": "frobnicate"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("unknown op")));
        let err = execute_tool("manageTodos", &json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("missing 'action'")));
    }

    #[tokio::test]
    async fn test_todos_corrupt_state_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        execute_tool(
            "writeFile",
            &json!({"path": "todos.json", "content": "not json {{{"}),
            &ctx,
        )
        .await
        .unwrap();
        let err = execute_tool("manageTodos", &json!({"op": "list"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("corrupt")));
        execute_tool(
            "writeFile",
            &json!({"path": "todos.json", "content": "{\"a\":1}", "overwrite": true}),
            &ctx,
        )
        .await
        .unwrap();
        let err = execute_tool("manageTodos", &json!({"op": "list"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(_)));
    }

    #[tokio::test]
    async fn test_todos_missing_file_lists_empty() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let out = execute_tool("manageTodos", &json!({"op": "list"}), &ctx).await.unwrap();
        assert!(out.contains("No tasks"));
    }

    #[tokio::test]
    async fn test_todos_file_stays_in_sandbox() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool(
            "manageTodos",
            &json!({"action": "list", "todoFile": "/tmp/evil-todos.json"}),
            &ctx,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)));
        execute_tool(
            "manageTodos",
            &json!({"action": "add", "text": "custom path", "todoFile": "custom/todos.json"}),
            &ctx,
        )
        .await
        .unwrap();
        let out = execute_tool(
            "manageTodos",
            &json!({"action": "list", "todoFile": "custom/todos.json"}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(out.contains("custom path"));
        assert!(dir.path().join("custom/todos.json").exists());
    }

    #[tokio::test]
    async fn test_load_skill_list_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let out = execute_tool("loadSkill", &json!({}), &ctx).await.unwrap();
        assert!(out.contains("commit-message"));
        assert!(out.contains("code-review-checklist"));
        let out = execute_tool("loadSkill", &json!({"action": "list"}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("commit-message"));
        let out = execute_tool("loadSkill", &json!({"name": "commit-message"}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("commit message"));
        let out = execute_tool(
            "loadSkill",
            &json!({"action": "load", "argument": "code-review-checklist"}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(out.contains("Checklist"));
        let out = execute_tool("loadSkill", &json!({"action": "search", "argument": "commit"}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("commit-message"));
    }

    #[tokio::test]
    async fn test_load_skill_unknown_name_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("loadSkill", &json!({"name": "no-such-skill"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("not found")));
        let err = execute_tool(
            "loadSkill",
            &json!({"action": "load", "argument": "no-such-skill"}),
            &ctx,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("not found")));
    }

    #[tokio::test]
    async fn test_shell_empty_command_fails() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = run_shell_checked(&ctx, "", 5).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(_)));
        let err = run_shell_checked(&ctx, "   ", 5).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(_)));
        let err = execute_tool("runShell", &json!({"command": ""}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(_)));
        let err = execute_tool("runShell", &json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("missing 'command'")));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_shell_timeout_kills_child() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = run_shell_checked(&ctx, "sleep 30", 1).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("timed out")));
        let err = execute_tool("runShell", &json!({"command": "sleep 30", "timeout_secs": 1}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("timed out")));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_shell_timeout_kills_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let pid_file = dir.path().join("child.pid");
        let cmd = format!(
            "(sh -c 'echo $$ > \"{}\"; exec sleep 30') & sleep 30",
            pid_file.display()
        );
        let res = run_shell_checked(&ctx, &cmd, 1).await;
        assert!(res.is_err());
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        if let Ok(pid_str) = std::fs::read_to_string(&pid_file)
            && let Ok(pid) = pid_str.trim().parse::<i32>()
        {
            let exists = unsafe { libc::kill(pid, 0) == 0 };
            assert!(
                !exists,
                "background child process in process group should have been killed"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_shell_env_keeps_path_drops_token() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let path_out = run_shell_checked(&ctx, "echo $PATH", 10).await.unwrap();
        assert_ne!(path_out.trim(), "");
        unsafe {
            std::env::set_var("POLYPORT_TASK4_PROBE_TOKEN", "probe-secret-abc123");
        }
        let env_out = run_shell_checked(&ctx, "env", 10).await.unwrap();
        unsafe {
            std::env::remove_var("POLYPORT_TASK4_PROBE_TOKEN");
        }
        assert!(!env_out.contains("probe-secret-abc123"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_shell_output_truncated_to_32kb() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let out = run_shell_checked(&ctx, "yes A | head -n 30000", 30).await.unwrap();
        assert!(out.contains('A'));
        assert!(out.len() <= 32 * 1024);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_shell_cwd_jailed_to_sandbox() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let out = run_shell_checked(&ctx, "pwd", 10).await.unwrap();
        let root = dir.path().canonicalize().unwrap();
        assert_eq!(out.trim(), root.to_string_lossy());
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
        assert!(!default_ctx.fetch_allow_loopback);

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

    #[tokio::test]
    async fn test_fetch_blocked_private_ip() {
        let ctx = ToolCtx::test_in(&tempfile::tempdir().unwrap());
        let err = execute_tool("fetchUrl", &json!({"url":"http://169.254.169.254/"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)));
    }

    #[tokio::test]
    async fn test_fetch_ssrf_literal_ranges_denied() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let blocked = [
            "http://127.0.0.1/",
            "http://10.0.0.5/",
            "http://172.16.0.9/",
            "http://172.31.255.1/",
            "http://192.168.0.1/",
            "http://100.64.0.1/",
            "http://192.0.2.1/",
            "http://198.51.100.2/",
            "http://203.0.113.3/",
            "http://198.18.0.1/",
            "http://224.0.0.1/",
            "http://255.255.255.255/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://[::]/",
            "http://[fe80::1]/",
            "http://[fc00::1]/",
            "http://[ff02::1]/",
            "http://[2001:db8::1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://localhost/",
            "http://LOCALHOST:8080/x",
            "ftp://127.0.0.1/x",
        ];
        for url in blocked {
            let err = execute_tool("fetchUrl", &json!({"url": url}), &ctx).await.unwrap_err();
            assert!(matches!(err, ToolError::Denied(_)), "expected Denied for {url}");
        }
    }

    async fn stub_base() -> String {
        use axum::{Router, response::Redirect, routing::get};
        let app = Router::new()
            .route("/ok", get(|| async { "hello stub" }))
            .route("/chain1", get(|| async { Redirect::to("/chain2") }))
            .route("/chain2", get(|| async { Redirect::to("/ok") }))
            .route("/k1", get(|| async { Redirect::to("/k2") }))
            .route("/k2", get(|| async { Redirect::to("/k3") }))
            .route("/k3", get(|| async { Redirect::to("/ok") }))
            .route("/loop", get(|| async { Redirect::to("/loop") }))
            .route("/evil", get(|| async { Redirect::to("http://10.255.255.1/") }))
            .route("/big", get(|| async { "A".repeat(200_000) }))
            .route(
                "/slow",
                get(|| async {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    "late"
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        base
    }

    fn stub_ctx(dir: &tempfile::TempDir) -> ToolCtx {
        let mut ctx = ToolCtx::test_in(dir);
        ctx.fetch_allow_loopback = true;
        ctx
    }

    #[tokio::test]
    async fn test_fetch_stub_ok_and_redirect_chain() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = stub_ctx(&dir);
        let base = stub_base().await;
        let out = execute_tool("fetchUrl", &json!({"url": format!("{base}/ok")}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("hello stub"));
        let out = execute_tool("fetchUrl", &json!({"url": format!("{base}/chain1")}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("hello stub"));
        let out = execute_tool("fetchUrl", &json!({"url": format!("{base}/k1")}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("hello stub"));
    }

    #[tokio::test]
    async fn test_fetch_redirect_loop_exceeds_max() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = stub_ctx(&dir);
        let base = stub_base().await;
        let err = execute_tool("fetchUrl", &json!({"url": format!("{base}/loop")}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("redirect")));
    }

    #[tokio::test]
    async fn test_fetch_redirect_to_private_denied() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = stub_ctx(&dir);
        let base = stub_base().await;
        let err = execute_tool("fetchUrl", &json!({"url": format!("{base}/evil")}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)));
    }

    #[tokio::test]
    async fn test_fetch_byte_cap_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = stub_ctx(&dir);
        ctx.fetch_max_bytes = 64;
        let base = stub_base().await;
        let out = execute_tool("fetchUrl", &json!({"url": format!("{base}/big")}), &ctx)
            .await
            .unwrap();
        assert_ne!(out, "");
        assert!(out.len() <= 64);
        assert!(out.chars().all(|ch| ch == 'A'));
    }

    #[tokio::test]
    async fn test_fetch_timeout_fails() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = stub_ctx(&dir);
        ctx.fetch_timeout_secs = 1;
        let base = stub_base().await;
        let err = execute_tool("fetchUrl", &json!({"url": format!("{base}/slow")}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("timed out")));
    }

    #[tokio::test]
    async fn test_fetch_non_2xx_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = stub_ctx(&dir);
        let base = stub_base().await;
        let err = execute_tool("fetchUrl", &json!({"url": format!("{base}/nope")}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("404")));
    }

    #[tokio::test]
    async fn test_search_empty_query_failed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::test_in(&dir);
        let err = execute_tool("searchWeb", &json!({"query": ""}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(_)));
        let err = execute_tool("searchWeb", &json!({"query": "   "}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failed(_)));
        let err = execute_tool("searchWeb", &json!({}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::Failed(ref msg) if msg.contains("missing 'query'")));
    }

    #[test]
    fn test_search_ddg_fixture_parse() {
        let raw = include_str!("../tests/fixtures/ddg.json");
        let data: serde_json::Value = serde_json::from_str(raw).unwrap();
        let out = format_ddg_results(&data, 5);
        assert_eq!(
            out,
            "Rust is a systems programming language focused on safety and performance.\n\
             Cargo - Rust package manager and build system (https://en.wikipedia.org/wiki/Cargo_(software))\n\
             The Rust Book - official guide (https://doc.rust-lang.org/book/)\n\
             Rust Playground - compile and run code in a browser (https://play.rust-lang.org/)\n\
             Rust official site (https://www.rust-lang.org/)"
        );
        let capped = format_ddg_results(&data, 2);
        assert_eq!(
            capped,
            "Rust is a systems programming language focused on safety and performance.\n\
             Cargo - Rust package manager and build system (https://en.wikipedia.org/wiki/Cargo_(software))"
        );
    }

    #[test]
    fn test_search_no_results() {
        let empty = serde_json::json!({"AbstractText": "", "RelatedTopics": []});
        assert_eq!(format_ddg_results(&empty, 5), "no results");
        let missing = serde_json::json!({});
        assert_eq!(format_ddg_results(&missing, 5), "no results");
    }
}
