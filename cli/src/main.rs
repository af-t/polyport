use clap::Parser;
use polyport_core::app::AppState;
use polyport_core::cli::{Cli, Commands, StartArgs, prepare_start, run_cli, serve_router};

fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn build_app(
    core_state: AppState,
    #[allow(unused)] port: u16,
    #[allow(unused)] proxy_only: bool,
    #[allow(unused)] sandbox_root: &std::path::Path,
) -> axum::Router {
    #[allow(unused_mut)]
    let mut app = polyport_core::app::create_app(core_state);

    #[cfg(feature = "agent")]
    if !proxy_only {
        app = app.merge(polyport_agent::ws::agent_router_with_root(
            port,
            sandbox_root.to_path_buf(),
        ));
    }

    #[cfg(feature = "webui")]
    {
        let ui_state = polyport_webui::UiState { asset_dir: None };
        let file_state = polyport_webui::FileState {
            sandbox_root: sandbox_root.to_path_buf(),
        };
        app = polyport_webui::mount_webui_routes(app, ui_state, file_state, proxy_only);
    }

    app
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let Cli { log_level, command } = cli;
    let command = command.unwrap_or(Commands::Start(StartArgs {
        port: None,
        proxy_only: false,
        sandbox: None,
        cache_ttl: None,
        log_level: None,
    }));

    match command {
        Commands::Start(args) => {
            let level = polyport_core::cli::effective_log_level(log_level.as_deref(), args.log_level.as_deref());
            polyport_core::logging::init_logging(level);
            let args = StartArgs {
                log_level: level.map(str::to_string),
                ..args
            };
            let port = prepare_start(&args).await?;
            let proxy_only = args.proxy_only || std::env::var("POLYPORT_PROXY_ONLY").is_ok_and(|v| v == "1");
            let sandbox_root = std::env::current_dir()?;
            let core_state = AppState::default();
            let app = build_app(core_state, port, proxy_only, &sandbox_root);
            serve_router(app, port).await?;
            Ok(())
        }
        other => {
            run_cli(Cli {
                log_level,
                command: Some(other),
            })
            .await?;
            Ok(())
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli).await {
        eprintln!("polyport {} error: {err:#}", version());
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::build_app;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use polyport_core::app::AppState;
    use tower::ServiceExt;

    #[test]
    fn binary_reports_crate_version() {
        assert_eq!(super::version(), env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_core_health_always_available() {
        let core_state = AppState::default();
        let sandbox_root = tempfile::tempdir().unwrap();
        let app = build_app(core_state, 9000, false, sandbox_root.path());
        let req = Request::builder().uri("/health").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[cfg(feature = "agent")]
    #[tokio::test]
    async fn test_chat_ws_excluded_when_proxy_only() {
        let core_state = AppState::default();
        let sandbox_root = tempfile::tempdir().unwrap();
        let app = build_app(core_state, 9000, true, sandbox_root.path());
        let req = Request::builder().uri("/api/chat").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(feature = "agent")]
    #[tokio::test]
    async fn test_chat_ws_mounted_when_not_proxy_only() {
        let core_state = AppState::default();
        let sandbox_root = tempfile::tempdir().unwrap();
        let app = build_app(core_state, 9000, false, sandbox_root.path());
        let req = Request::builder().uri("/api/chat").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_ne!(res.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(feature = "webui")]
    #[tokio::test]
    async fn test_webui_file_endpoint_mounted() {
        let core_state = AppState::default();
        let sandbox_root = tempfile::tempdir().unwrap();
        let app = build_app(core_state, 9000, false, sandbox_root.path());
        let req = Request::builder().uri("/api/file?path=").body(Body::empty()).unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
