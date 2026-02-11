use std::env;
use std::path::Path;

use crate::config::DaemonConfig;

pub struct CliInvocation {
    pub config: DaemonConfig,
    pub command: String,
    pub command_args: Vec<String>,
}

pub fn print_usage() {
    eprintln!("Ferrite-OS Daemon");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    ferrite-daemon [serve|start] [OPTIONS]");
    eprintln!("    ferrite-daemon <COMMAND> [OPTIONS]");
    eprintln!();
    eprintln!("SERVER COMMANDS:");
    eprintln!("    serve, start          Start the daemon");
    eprintln!();
    eprintln!("CLIENT COMMANDS:");
    eprintln!("    ping                  Test connectivity");
    eprintln!("    status                Get pool status");
    eprintln!("    stats                 Get runtime statistics");
    eprintln!("    metrics               Get comprehensive metrics");
    eprintln!("    snapshot              Get system snapshot");
    eprintln!("    health                Get health check");
    eprintln!("    keepalive             Send keepalive");
    eprintln!("    apps                  List managed apps");
    eprintln!("    app-start APP [ARGS]  Start managed app");
    eprintln!("    app-stop ID|NAME      Stop managed app");
    eprintln!();
    eprintln!("JOB COMMANDS:");
    eprintln!("    job-submit CMD [ARGS] Submit a durable job");
    eprintln!("    job-stop ID [REASON]  Cancel a durable job");
    eprintln!("    job-status ID         Show job status");
    eprintln!("    job-list              List all durable jobs");
    eprintln!("    job-history ID        Show job state transitions");
    eprintln!();
    eprintln!("OTHER:");
    eprintln!("    shutdown              Shutdown daemon");
    eprintln!("    watch                 Watch metrics (live)");
    eprintln!("    help                  Show help");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("    --config FILE         Load configuration from file");
    eprintln!("    --socket PATH         Unix socket path");
    eprintln!("    --device N            GPU device ID");
    eprintln!("    --streams N           Maximum streams");
    eprintln!("    --pool-fraction F     VRAM pool fraction (0.0-1.0)");
    eprintln!("    --boot-kernel         Boot persistent kernel");
    eprintln!("    --watch               Enable watch mode");
    eprintln!("    --watch-ms N          Watch interval (milliseconds)");
    eprintln!("    --log-dir DIR         Log directory");
    eprintln!("    --apps-bin-dir DIR    Managed app binaries directory");
    eprintln!("    --headless            Skip TUI, run headless");
    eprintln!("    --gpu-name NAME       GPU device name for TUI header");
    eprintln!();
    eprintln!("ENVIRONMENT:");
    eprintln!("    FERRITE_DEVICE           GPU device ID");
    eprintln!("    FERRITE_SOCKET           Socket path");
    eprintln!("    FERRITE_MAX_STREAMS      Maximum streams");
    eprintln!("    FERRITE_BOOT_KERNEL      Boot persistent kernel");
    eprintln!("    FERRITE_WATCH            Enable watch mode");
    eprintln!("    FERRITE_APPS_BIN_DIR     Managed app binaries directory");
    eprintln!("    FERRITE_GPU_NAME         GPU device name for TUI header");
    eprintln!("    FERRITE_HEADLESS         Skip TUI, run headless");
}

pub fn parse_cli() -> Result<CliInvocation, i32> {
    let mut config = DaemonConfig::default();
    let mut command: Option<String> = None;
    let mut command_args: Vec<String> = Vec::new();
    let mut args = env::args().skip(1).peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                if let Some(path) = args.next() {
                    match DaemonConfig::load_from_file(Path::new(&path)) {
                        Ok(loaded) => config = loaded,
                        Err(e) => {
                            eprintln!("Error loading config: {}", e);
                            return Err(1);
                        }
                    }
                } else {
                    eprintln!("--config requires a value");
                    return Err(2);
                }
            }
            "--socket" => {
                config.socket_path = args.next().expect("--socket requires a value");
            }
            "--device" => {
                config.device_id = args
                    .next()
                    .expect("--device requires a value")
                    .parse()
                    .expect("Invalid device ID");
            }
            "--streams" => {
                config.max_streams = args
                    .next()
                    .expect("--streams requires a value")
                    .parse()
                    .expect("Invalid streams value");
            }
            "--pool-fraction" => {
                config.pool_fraction = args
                    .next()
                    .expect("--pool-fraction requires a value")
                    .parse()
                    .expect("Invalid pool fraction");
            }
            "--boot-kernel" => {
                config.boot_kernel = true;
            }
            "--watch" => {
                config.watch_enabled = true;
            }
            "--watch-ms" => {
                config.watch_ms = args
                    .next()
                    .expect("--watch-ms requires a value")
                    .parse()
                    .expect("Invalid watch interval");
            }
            "--log-dir" => {
                config.log_dir = Some(args.next().expect("--log-dir requires a value"));
            }
            "--apps-bin-dir" => {
                config.apps_bin_dir = Some(args.next().expect("--apps-bin-dir requires a value"));
            }
            "--headless" => {
                config.headless = true;
            }
            "--gpu-name" => {
                config.gpu_name = Some(args.next().expect("--gpu-name requires a value"));
            }
            val if !val.starts_with("--") => {
                if command.is_none() {
                    command = Some(val.to_string());
                } else {
                    command_args.push(val.to_string());
                }
            }
            _ => {
                eprintln!("Unknown option: {}", arg);
                return Err(2);
            }
        }
    }

    config.merge_from_env();

    Ok(CliInvocation {
        config,
        command: command.unwrap_or_else(|| "serve".to_string()),
        command_args,
    })
}
