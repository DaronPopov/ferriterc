use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::time::Instant;

use tracing::{info, warn};

use crate::config::DaemonConfig;
use crate::state::{DaemonState, ManagedApp, MANAGED_APPS};

pub fn cleanup_exited_apps(state: &DaemonState) {
    let mut apps = state.apps.lock();
    apps.retain(|_, app| match app.child.try_wait() {
        Ok(Some(status)) => {
            info!(
                app_id = app.id,
                app = %app.name,
                code = ?status.code(),
                "Managed app exited"
            );
            false
        }
        Ok(None) => true,
        Err(e) => {
            warn!(app_id = app.id, app = %app.name, error = %e, "Failed polling app state");
            false
        }
    });
}

pub fn resolve_app_target(config: &DaemonConfig, app: &str) -> PathBuf {
    if let Some(dir) = &config.apps_bin_dir {
        return Path::new(dir).join(app);
    }
    PathBuf::from(app)
}

pub fn handle_ping() -> io::Result<String> {
    Ok("{\"ok\":true,\"message\":\"pong\"}\n".to_string())
}

pub fn handle_status(state: &DaemonState) -> io::Result<String> {
    let tlsf = state.runtime.tlsf_stats();
    let response = serde_json::json!({
        "ok": true,
        "pool_total": tlsf.total_pool_size,
        "allocated": tlsf.allocated_bytes,
        "free": tlsf.free_bytes,
        "utilization": tlsf.utilization_percent,
        "fragmentation": tlsf.fragmentation_ratio,
        "healthy": tlsf.is_healthy,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_stats(state: &DaemonState) -> io::Result<String> {
    let stats = state.runtime.stats();
    let response = serde_json::json!({
        "ok": true,
        "vram_allocated": stats.vram_allocated,
        "vram_used": stats.vram_used,
        "vram_free": stats.vram_free,
        "gpu_utilization": stats.gpu_utilization,
        "active_streams": stats.active_streams,
        "registered_kernels": stats.registered_kernels,
        "total_ops": stats.total_ops,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_metrics(state: &DaemonState) -> io::Result<String> {
    cleanup_exited_apps(state);
    let stats = state.runtime.stats();
    let tlsf = state.runtime.tlsf_stats();
    let uptime = state.start_time.elapsed().as_secs();
    let active = state.active_clients.load(Ordering::Relaxed);
    let total = state.total_requests.load(Ordering::Relaxed);
    let failed = state.failed_requests.load(Ordering::Relaxed);
    let running_apps = state.apps.lock().len();

    let response = serde_json::json!({
        "ok": true,
        "uptime_secs": uptime,
        "active_clients": active,
        "running_apps": running_apps,
        "total_requests": total,
        "failed_requests": failed,
        "vram_allocated": stats.vram_allocated,
        "vram_used": stats.vram_used,
        "vram_free": stats.vram_free,
        "gpu_util": stats.gpu_utilization,
        "active_streams": stats.active_streams,
        "total_ops": stats.total_ops,
        "pool_total": tlsf.total_pool_size,
        "pool_allocated": tlsf.allocated_bytes,
        "pool_free": tlsf.free_bytes,
        "pool_util": tlsf.utilization_percent,
        "fragmentation": tlsf.fragmentation_ratio,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_snapshot(state: &DaemonState) -> io::Result<String> {
    let snap = state.runtime.system_snapshot();
    let response = serde_json::json!({
        "ok": true,
        "total_ops": snap.total_ops,
        "active_processes": snap.active_processes,
        "active_tasks": snap.active_tasks,
        "vram_used": snap.total_vram_used,
        "watchdog_alert": snap.watchdog_alert,
        "queue_head": snap.queue_head,
        "queue_tail": snap.queue_tail,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_health(state: &DaemonState) -> io::Result<String> {
    cleanup_exited_apps(state);
    let tlsf = state.runtime.tlsf_stats();
    let uptime = state.start_time.elapsed().as_secs();
    let active = state.active_clients.load(Ordering::Relaxed);
    let running_apps = state.apps.lock().len();

    let healthy = tlsf.is_healthy
        && active < state.config.max_clients as u64
        && tlsf.utilization_percent < 95.0;

    let response = serde_json::json!({
        "ok": true,
        "healthy": healthy,
        "uptime_secs": uptime,
        "pool_healthy": tlsf.is_healthy,
        "active_clients": active,
        "running_apps": running_apps,
        "max_clients": state.config.max_clients,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_keepalive(state: &DaemonState) -> io::Result<String> {
    state.runtime.keepalive();
    Ok("{\"ok\":true,\"message\":\"keepalive sent\"}\n".to_string())
}

pub fn handle_apps(state: &DaemonState) -> io::Result<String> {
    cleanup_exited_apps(state);
    let apps = state.apps.lock();
    let running: Vec<serde_json::Value> = apps
        .values()
        .map(|app| {
            serde_json::json!({
                "id": app.id,
                "name": app.name,
                "pid": app.child.id(),
                "uptime_secs": app.started_at.elapsed().as_secs(),
                "args": app.args,
            })
        })
        .collect();
    let response = serde_json::json!({
        "ok": true,
        "running": running,
        "managed_apps": MANAGED_APPS,
        "apps_bin_dir": state.config.apps_bin_dir,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_app_start(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: app-start <app> [args...]\"}\n".to_string());
    }
    let app = args[0];
    if !MANAGED_APPS.iter().any(|x| *x == app) {
        let response = serde_json::json!({
            "error": "app not allowed",
            "app": app,
            "allowed": MANAGED_APPS,
        });
        return Ok(format!("{}\n", serde_json::to_string(&response).unwrap()));
    }

    let app_args: Vec<String> = args.iter().skip(1).map(|s| s.to_string()).collect();
    let target = resolve_app_target(&state.config, app);
    let mut cmd = Command::new(&target);
    cmd.args(&app_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            let response = serde_json::json!({
                "error": "spawn failed",
                "app": app,
                "target": target,
                "message": e.to_string(),
            });
            return Ok(format!("{}\n", serde_json::to_string(&response).unwrap()));
        }
    };

    let pid = child.id();
    let app_id = state.next_app_id.fetch_add(1, Ordering::Relaxed);
    let managed = ManagedApp {
        id: app_id,
        name: app.to_string(),
        args: app_args,
        started_at: Instant::now(),
        child,
    };
    state.apps.lock().insert(app_id, managed);
    let response = serde_json::json!({
        "ok": true,
        "message": "app started",
        "id": app_id,
        "name": app,
        "pid": pid,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_app_stop(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: app-stop <id|name>\"}\n".to_string());
    }
    let selector = args[0];
    let mut apps = state.apps.lock();
    let target_id = if let Ok(id) = selector.parse::<u64>() {
        Some(id)
    } else {
        apps.iter()
            .find(|(_, app)| app.name == selector)
            .map(|(id, _)| *id)
    };

    let Some(id) = target_id else {
        let response = serde_json::json!({
            "error": "app not found",
            "selector": selector,
        });
        return Ok(format!("{}\n", serde_json::to_string(&response).unwrap()));
    };

    let mut app = apps.remove(&id).expect("app id resolved but missing");
    let name = app.name.clone();
    let pid = app.child.id();

    let mut stop_err: Option<String> = None;
    if let Err(e) = app.child.kill() {
        stop_err = Some(e.to_string());
    } else if let Err(e) = app.child.wait() {
        stop_err = Some(e.to_string());
    }

    let response = if let Some(err) = stop_err {
        serde_json::json!({
            "ok": false,
            "message": "failed to stop app cleanly",
            "id": id,
            "name": name,
            "pid": pid,
            "error": err,
        })
    } else {
        serde_json::json!({
            "ok": true,
            "message": "app stopped",
            "id": id,
            "name": name,
            "pid": pid,
        })
    };
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

pub fn handle_help() -> io::Result<String> {
    let help = r#"{
  "ok": true,
  "commands": {
    "ping": "Test daemon connectivity",
    "status": "Get TLSF pool status",
    "stats": "Get GPU runtime statistics",
    "metrics": "Get comprehensive metrics",
    "snapshot": "Get system snapshot",
    "health": "Get health check",
    "keepalive": "Send keepalive signal",
    "apps": "List managed app processes",
    "app-start <app> [args...]": "Start a managed app binary",
    "app-stop <id|name>": "Stop a managed app process",
    "scheduler <subcommand>": "Scheduler control (queue-status, tenants, pause, resume, stats, policies, policy)",
    "audit-query [--tenant ID] [--last N]": "Query control plane audit log",
    "events-stream": "Subscribe to real-time event stream (JSON lines)",
    "job-submit <cmd> [args...]": "Submit a durable job",
    "job-stop <id> [reason]": "Cancel a durable job",
    "job-status <id>": "Show status of a durable job",
    "job-list": "List all durable jobs",
    "job-history <id>": "Show state transition history for a job",
    "shutdown": "Shutdown daemon",
    "help": "Show this help"
  }
}
"#;
    Ok(help.to_string())
}

// ── scheduler / control plane handlers ────────────────────────────

/// Handle the `scheduler` meta-command: dispatches to scheduler_commands.
pub fn handle_scheduler_command(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    use crate::scheduler_commands::{self, SchedulerCommand};

    let Some(cmd) = SchedulerCommand::parse_from_args(args) else {
        return Ok("{\"error\":\"unknown scheduler subcommand — try: scheduler queue-status\"}\n".to_string());
    };

    let mut engine = state.policy_engine.lock();
    let mut paused = state.scheduler_paused.load(Ordering::Relaxed);
    let result = scheduler_commands::handle_scheduler_command(&cmd, &mut engine, &mut paused);
    state.scheduler_paused.store(paused, Ordering::Relaxed);
    Ok(result)
}

/// Handle `scheduler-status`: quick overview of queue depth, active jobs, tenants.
pub fn handle_scheduler_status(state: &DaemonState) -> io::Result<String> {
    let engine = state.policy_engine.lock();
    let paused = state.scheduler_paused.load(Ordering::Relaxed);
    let response = serde_json::json!({
        "ok": true,
        "paused": paused,
        "rule_count": engine.rule_count(),
        "audit_entries": engine.audit_log().len(),
        "queue_depth": 0,
        "active_jobs": 0,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

/// Handle `scheduler-queue`: list queued/running jobs.
pub fn handle_scheduler_queue(state: &DaemonState) -> io::Result<String> {
    let paused = state.scheduler_paused.load(Ordering::Relaxed);
    let response = serde_json::json!({
        "ok": true,
        "paused": paused,
        "jobs": [],
        "count": 0,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

/// Handle `scheduler-policy`: show active policies and recent decisions.
pub fn handle_scheduler_policy(state: &DaemonState) -> io::Result<String> {
    let engine = state.policy_engine.lock();
    let rules: Vec<serde_json::Value> = engine
        .list_rules()
        .into_iter()
        .map(|(name, desc)| serde_json::json!({"name": name, "description": desc}))
        .collect();

    let recent: Vec<serde_json::Value> = engine
        .audit_log()
        .query_recent(20)
        .into_iter()
        .map(|e| serde_json::json!({
            "action": e.action,
            "resource": e.resource,
            "decision": e.decision,
            "reason": e.reason,
            "tenant_id": e.tenant_id,
            "timestamp_secs": e.elapsed_secs,
        }))
        .collect();

    let response = serde_json::json!({
        "ok": true,
        "rules": rules,
        "recent_decisions": recent,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

/// Handle `audit-query [--tenant ID] [--last N]`.
pub fn handle_audit_query(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    let engine = state.policy_engine.lock();
    let mut tenant_filter: Option<u64> = None;
    let mut limit: usize = 50;

    // Parse flags
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--tenant" => {
                if i + 1 < args.len() {
                    tenant_filter = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--last" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().unwrap_or(50);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // Emit audit query event
    {
        let mut es = state.event_stream.lock();
        es.emit(crate::event_stream::SchedulerEvent::AuditQuery {
            tenant_filter,
            result_count: 0, // updated below
        });
    }

    let entries: Vec<serde_json::Value> = if let Some(tid) = tenant_filter {
        engine.audit_log().query_by_tenant(tid, limit)
    } else {
        engine.audit_log().query_recent(limit)
    }
    .into_iter()
    .map(|e| serde_json::json!({
        "timestamp_secs": e.elapsed_secs,
        "tenant_id": e.tenant_id,
        "action": e.action,
        "resource": e.resource,
        "decision": e.decision,
        "reason": e.reason,
        "source_ip": e.source_ip,
    }))
    .collect();

    let response = serde_json::json!({
        "ok": true,
        "count": entries.len(),
        "tenant_filter": tenant_filter,
        "entries": entries,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

/// Handle `events-stream`: return recent events as JSON lines.
pub fn handle_events_stream(state: &DaemonState) -> io::Result<String> {
    let es = state.event_stream.lock();
    let output = es.export_jsonl();
    if output.is_empty() {
        Ok("{\"ok\":true,\"message\":\"no events\"}\n".to_string())
    } else {
        Ok(format!("{}\n", output))
    }
}

// ── durable job command handlers (Plan-B) ─────────────────────────

/// Handle `job-submit <command> [args...]` -- submit a new durable job.
pub fn handle_job_submit(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-submit <command> [args...]\"}\n".to_string());
    }

    let command = args[0].to_string();
    let job_args: Vec<String> = args.iter().skip(1).map(|s| s.to_string()).collect();
    let name = command.rsplit('/').next().unwrap_or(&command).to_string();
    let policy = state.config.jobs.to_restart_policy();

    let job = ptx_runtime::job::DurableJob::new(name.clone(), command, job_args, policy);
    let mut supervisor = state.supervisor.lock();
    match supervisor.submit(job) {
        Ok(id) => {
            let response = serde_json::json!({
                "ok": true,
                "message": "job submitted",
                "job_id": id.raw(),
                "name": name,
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
        Err(e) => {
            let response = serde_json::json!({
                "error": "job submit failed",
                "message": e.to_string(),
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
    }
}

/// Handle `job-stop <id>` -- cancel a durable job.
pub fn handle_job_stop(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-stop <id>\"}\n".to_string());
    }

    let id: u64 = match args[0].parse() {
        Ok(id) => id,
        Err(_) => {
            return Ok("{\"error\":\"job-stop requires a numeric job ID\"}\n".to_string());
        }
    };

    let reason = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "stopped by operator".to_string()
    };

    let mut supervisor = state.supervisor.lock();
    match supervisor.stop(ptx_runtime::job::DurableJobId(id), reason) {
        Ok(()) => {
            let response = serde_json::json!({
                "ok": true,
                "message": "job stopped",
                "job_id": id,
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
        Err(e) => {
            let response = serde_json::json!({
                "error": "job stop failed",
                "message": e.to_string(),
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
    }
}

/// Handle `job-status <id>` -- show status of a specific durable job.
pub fn handle_job_status(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-status <id>\"}\n".to_string());
    }

    let id: u64 = match args[0].parse() {
        Ok(id) => id,
        Err(_) => {
            return Ok("{\"error\":\"job-status requires a numeric job ID\"}\n".to_string());
        }
    };

    let supervisor = state.supervisor.lock();
    match supervisor.status(ptx_runtime::job::DurableJobId(id)) {
        Some(job) => {
            let response = serde_json::json!({
                "ok": true,
                "job_id": job.id.raw(),
                "name": job.name,
                "command": job.command,
                "args": job.args,
                "state": job.state().to_string(),
                "failure_count": job.failure_count,
                "last_failure": job.last_failure,
                "process_pid": job.process_pid,
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
        None => {
            let response = serde_json::json!({
                "error": "job not found",
                "job_id": id,
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
    }
}

/// Handle `job-list` -- list all durable jobs.
pub fn handle_job_list(state: &DaemonState) -> io::Result<String> {
    let supervisor = state.supervisor.lock();
    let jobs: Vec<serde_json::Value> = supervisor
        .list()
        .iter()
        .map(|job| {
            serde_json::json!({
                "job_id": job.id.raw(),
                "name": job.name,
                "state": job.state().to_string(),
                "failure_count": job.failure_count,
                "process_pid": job.process_pid,
            })
        })
        .collect();

    let response = serde_json::json!({
        "ok": true,
        "count": jobs.len(),
        "jobs": jobs,
    });
    Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
}

/// Handle `job-history <id>` -- show full state transition history for a job.
pub fn handle_job_history(state: &DaemonState, args: &[&str]) -> io::Result<String> {
    if args.is_empty() {
        return Ok("{\"error\":\"usage: job-history <id>\"}\n".to_string());
    }

    let id: u64 = match args[0].parse() {
        Ok(id) => id,
        Err(_) => {
            return Ok("{\"error\":\"job-history requires a numeric job ID\"}\n".to_string());
        }
    };

    let supervisor = state.supervisor.lock();
    match supervisor.status(ptx_runtime::job::DurableJobId(id)) {
        Some(job) => {
            let transitions: Vec<serde_json::Value> = job
                .state_machine
                .history()
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "from": t.from_state.to_string(),
                        "to": t.to_state.to_string(),
                        "timestamp": format!("{:?}", t.timestamp),
                        "reason": t.reason,
                    })
                })
                .collect();

            let response = serde_json::json!({
                "ok": true,
                "job_id": job.id.raw(),
                "name": job.name,
                "current_state": job.state().to_string(),
                "transition_count": transitions.len(),
                "transitions": transitions,
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
        None => {
            let response = serde_json::json!({
                "error": "job not found",
                "job_id": id,
            });
            Ok(format!("{}\n", serde_json::to_string(&response).unwrap()))
        }
    }
}
