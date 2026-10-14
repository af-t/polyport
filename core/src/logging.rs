use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Off,
}

impl LogLevel {
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "trace" => Self::Trace,
            "debug" => Self::Debug,
            "warn" | "warning" => Self::Warn,
            "error" => Self::Error,
            "silent" | "off" => Self::Off,
            _ => Self::Info,
        }
    }

    #[must_use]
    pub const fn as_filter_str(&self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Off => "off",
        }
    }
}

#[must_use]
pub fn resolve_filter_directive(explicit_level: Option<&str>) -> String {
    if let Some(level) = explicit_level {
        return LogLevel::parse(level).as_filter_str().to_string();
    }
    if let Ok(val) = std::env::var("LOG_LEVEL")
        && !val.trim().is_empty()
    {
        return LogLevel::parse(&val).as_filter_str().to_string();
    }
    if let Ok(val) = std::env::var("RUST_LOG")
        && !val.trim().is_empty()
    {
        return val;
    }
    "info".to_string()
}

pub fn init_logging(explicit_level: Option<&str>) {
    let directive = resolve_filter_directive(explicit_level);
    let filter = EnvFilter::new(&directive);

    let format_json = std::env::var("LOG_FORMAT").is_ok_and(|v| v.eq_ignore_ascii_case("json"));

    if format_json {
        let subscriber = tracing_subscriber::fmt().json().with_env_filter(filter).finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    } else {
        let subscriber = tracing_subscriber::fmt().with_env_filter(filter).finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_parsing() {
        assert_eq!(LogLevel::parse("trace"), LogLevel::Trace);
        assert_eq!(LogLevel::parse("debug"), LogLevel::Debug);
        assert_eq!(LogLevel::parse("DEBUG"), LogLevel::Debug);
        assert_eq!(LogLevel::parse("info"), LogLevel::Info);
        assert_eq!(LogLevel::parse("warn"), LogLevel::Warn);
        assert_eq!(LogLevel::parse("warning"), LogLevel::Warn);
        assert_eq!(LogLevel::parse("error"), LogLevel::Error);
        assert_eq!(LogLevel::parse("silent"), LogLevel::Off);
        assert_eq!(LogLevel::parse("off"), LogLevel::Off);
        assert_eq!(LogLevel::parse("unknown_level"), LogLevel::Info);
    }

    #[test]
    fn test_resolve_filter_directive() {
        assert_eq!(resolve_filter_directive(Some("debug")), "debug");
        assert_eq!(resolve_filter_directive(Some("silent")), "off");
        assert_eq!(resolve_filter_directive(Some("invalid")), "info");
    }

    #[test]
    fn test_init_logging_idempotent() {
        init_logging(Some("debug"));
        init_logging(Some("info"));
    }
}
