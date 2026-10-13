#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Gemini,
    Claude,
    Codex,
}

impl Provider {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Gemini => "gemini",
            Self::Claude => "claude",
            Self::Codex => "codex",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelClass {
    Claude,
    GeminiNative,
    Codex,
}
