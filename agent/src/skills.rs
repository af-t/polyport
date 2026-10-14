#[must_use]
pub fn skill_names() -> Vec<&'static str> {
    vec!["code-review-checklist", "commit-message"]
}

#[must_use]
pub fn load_skill(name: &str) -> Option<&'static str> {
    match name {
        "code-review-checklist" => Some(include_str!("../skills/code-review-checklist.md")),
        "commit-message" => Some(include_str!("../skills/commit-message.md")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_names_lists_both() {
        assert_eq!(skill_names(), vec!["code-review-checklist", "commit-message"]);
    }

    #[test]
    fn test_load_skill_returns_markdown() {
        let body = load_skill("commit-message").unwrap();
        assert!(body.contains("commit message"));
        let body = load_skill("code-review-checklist").unwrap();
        assert!(body.contains("Checklist"));
    }

    #[test]
    fn test_load_skill_unknown_is_none() {
        assert_eq!(load_skill("no-such-skill"), None);
    }
}
