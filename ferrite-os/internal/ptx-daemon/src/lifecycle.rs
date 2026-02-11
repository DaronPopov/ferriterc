use crate::bootstrap::CliInvocation;
use crate::server;

pub fn dispatch(invocation: CliInvocation) -> Result<(), i32> {
    match invocation.command.as_str() {
        "serve" | "start" => {
            if let Err(e) = server::run_server(invocation.config) {
                eprintln!("Daemon error: {}", e);
                return Err(1);
            }
            Ok(())
        }
        "watch" => {
            if let Err(e) = server::run_watch_client(&invocation.config.socket_path, invocation.config.watch_ms) {
                eprintln!("Watch error: {}", e);
                return Err(1);
            }
            Ok(())
        }
        cmd @ ("ping" | "status" | "stats" | "metrics" | "snapshot" | "health" | "keepalive"
        | "shutdown" | "help" | "apps" | "app-start" | "app-stop"
        | "job-submit" | "job-stop" | "job-status" | "job-list" | "job-history") => {
            let mut line = cmd.to_string();
            for arg in &invocation.command_args {
                line.push(' ');
                line.push_str(arg);
            }
            if let Err(e) = server::connect_and_send(&invocation.config.socket_path, &line) {
                eprintln!("Command error: {}", e);
                return Err(1);
            }
            Ok(())
        }
        _ => Err(2),
    }
}
