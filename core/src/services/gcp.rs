use std::path::{Path, PathBuf};

#[must_use]
pub fn get_gcp_project_id() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let share_dir = PathBuf::from(home).join(".local/share/cli-proxy-api");
    if share_dir.is_dir()
        && let Ok(entries) = std::fs::read_dir(share_dir)
    {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("antigravity-")
                && Path::new(&name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
                && let Ok(content) = std::fs::read_to_string(entry.path())
                && let Ok(val) = serde_json::from_str::<serde_json::Value>(&content)
                && let Some(pid) = val.get("project_id").and_then(serde_json::Value::as_str)
            {
                return pid.to_string();
            }
        }
    }
    "primal-pointer-x4gkq".to_string()
}
