fn main() {
    println!("cargo::rerun-if-changed=public");
    assert!(
        std::path::Path::new("public").is_dir(),
        "public/ not found: build the Web UI first with `bun run build:frontend`"
    );
}
