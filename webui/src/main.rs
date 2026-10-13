use clap::Parser;
use polyport_core::cli::{Cli, run_cli};

fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(err) = run_cli(cli).await {
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
