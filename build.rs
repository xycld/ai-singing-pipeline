fn main() {
    let world_src = "vendor/World/src";

    let cpp_files: Vec<String> = [
        "cheaptrick",
        "codec",
        "common",
        "d4c",
        "dio",
        "fft",
        "harvest",
        "matlabfunctions",
        "stonemask",
        "synthesis",
        "synthesisrealtime",
    ]
    .iter()
    .map(|name| format!("{world_src}/{name}.cpp"))
    .collect();

    cc::Build::new()
        .cpp(true)
        .include(world_src)
        .opt_level(2)
        .warnings(false)
        .files(&cpp_files)
        .compile("world");
}
