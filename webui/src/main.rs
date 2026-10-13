fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn main() {
    println!("polyport {}", version());
}

#[cfg(test)]
mod tests {
    #[test]
    fn binary_reports_crate_version() {
        assert_eq!(super::version(), env!("CARGO_PKG_VERSION"));
    }
}
