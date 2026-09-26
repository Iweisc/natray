mod app;
mod balloon;
mod cache;
mod capture;
mod env;
mod flight;
mod gpu;
mod lip;
mod params;
mod presets;
mod render;
mod sim;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--capture") {
        return capture::run(&args);
    }
    app::run()
}
