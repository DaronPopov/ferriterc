use std::io;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::commands;
use crate::state::DaemonState;

pub(super) struct ParsedCommand {
    pub command: String,
    pub args: Vec<String>,
}

pub(super) fn parse_command_line(cmdline: &str) -> Option<ParsedCommand> {
    let mut parts = cmdline.split_whitespace();
    let command = parts.next().unwrap_or("").to_lowercase();
    if command.is_empty() {
        return None;
    }
    let args = parts.map(|s| s.to_string()).collect();
    Some(ParsedCommand { command, args })
}

pub(super) fn execute_command(
    state: &Arc<DaemonState>,
    parsed: &ParsedCommand,
) -> io::Result<String> {
    let args_refs: Vec<&str> = parsed.args.iter().map(String::as_str).collect();

    match parsed.command.as_str() {
        "ping" => commands::handle_ping(),
        "status" => commands::handle_status(state),
        "stats" => commands::handle_stats(state),
        "metrics" => commands::handle_metrics(state),
        "snapshot" => commands::handle_snapshot(state),
        "health" => commands::handle_health(state),
        "keepalive" => commands::handle_keepalive(state),
        "apps" => commands::handle_apps(state),
        "app-start" => commands::handle_app_start(state, &args_refs),
        "app-stop" => commands::handle_app_stop(state, &args_refs),
        "shutdown" => {
            state.shutdown();
            Ok("{\"ok\":true,\"message\":\"shutting down\"}\n".to_string())
        }
        "help" => commands::handle_help(),
        "agent" => {
            // Agent protocol: parse JSON from first arg, dispatch to TUI
            let json_str = parsed.args.join(" ");
            if json_str.is_empty() {
                return Ok("{\"error\":\"usage: agent {\\\"cmd\\\":\\\"Status\\\"}\"}\n".to_string());
            }
            match serde_json::from_str::<crate::tui::agent::protocol::AgentCommand>(&json_str) {
                Ok(_cmd) => {
                    // In the socket context we don't have direct TUI access.
                    // The agent command needs to go through the event channel.
                    // For now, return a helpful message.
                    Ok("{\"ok\":true,\"message\":\"agent command parsed — use TUI event channel for execution\"}\n".to_string())
                }
                Err(e) => {
                    Ok(format!("{{\"error\":\"invalid agent command: {}\"}}\n", e))
                }
            }
        }

        // ── scheduler / control plane commands ─────────────────
        "scheduler" => commands::handle_scheduler_command(state, &args_refs),
        "scheduler-status" => commands::handle_scheduler_status(state),
        "scheduler-queue" => commands::handle_scheduler_queue(state),
        "scheduler-policy" => commands::handle_scheduler_policy(state),
        "audit-query" => commands::handle_audit_query(state, &args_refs),
        "events-stream" => commands::handle_events_stream(state),

        // ── durable job commands (Plan-B) ──────────────────────
        "job-submit" => commands::handle_job_submit(state, &args_refs),
        "job-stop" => commands::handle_job_stop(state, &args_refs),
        "job-status" => commands::handle_job_status(state, &args_refs),
        "job-list" => commands::handle_job_list(state),
        "job-history" => commands::handle_job_history(state, &args_refs),

        _ => {
            state.failed_requests.fetch_add(1, Ordering::Relaxed);
            Ok(format!(
                "{{\"error\":\"unknown command\",\"command\":\"{}\"}}\n",
                parsed.command
            ))
        }
    }
}
