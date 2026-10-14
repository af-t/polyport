use clap::Parser;
use polyport_core::app::AppState;
use polyport_core::cli::{Cli, Commands, StartArgs, prepare_start, run_cli, serve_router};

mod file_api;
mod serve;
mod static_files;

use file_api::FileState;
use serve::build_webui_app;
use static_files::UiState;

fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let Cli { log_level, command } = cli;
    match command {
        None => {
            polyport_core::logging::init_logging(log_level.as_deref());
            let args = StartArgs {
                port: None,
                proxy_only: false,
                sandbox: None,
                cache_ttl: None,
                log_level,
            };
            let port = prepare_start(&args).await?;
            let proxy_only = args.proxy_only || std::env::var("POLYPORT_PROXY_ONLY").is_ok_and(|v| v == "1");
            let sandbox_root = std::env::current_dir()?;
            let core_state = AppState::default();
            let ui_state = UiState { asset_dir: None };
            let file_state = FileState { sandbox_root };
            let app = build_webui_app(core_state, ui_state, file_state, proxy_only);
            serve_router(app, port).await?;
            Ok(())
        }
        Some(Commands::Start(args)) => {
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
            let ui_state = UiState { asset_dir: None };
            let file_state = FileState { sandbox_root };
            let app = build_webui_app(core_state, ui_state, file_state, proxy_only);
            serve_router(app, port).await?;
            Ok(())
        }
        other => {
            run_cli(Cli {
                log_level,
                command: other,
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
    #[test]
    fn binary_reports_crate_version() {
        assert_eq!(super::version(), env!("CARGO_PKG_VERSION"));
    }
}
