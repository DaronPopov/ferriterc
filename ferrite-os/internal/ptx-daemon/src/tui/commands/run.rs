use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use crate::events::{DaemonEvent, LogCategory, LogEntry};
use crate::script_runner::ScriptRunner;
use crate::tui::state::run_state::{RunOutputLine, RunProfile, RunStatus};
use crate::tui::state::TuiState;

pub fn cmd_run(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) {
    if state.run_status != RunStatus::Idle {
        state.push_log(LogEntry::new(
            LogCategory::Err,
            "already running — use 'stop' first",
        ));
        return;
    }

    let script = if state.file_dirty || state.open_file.is_some() {
        state.file_lines.join("\n")
    } else {
        state.push_log(LogEntry::new(LogCategory::Err, "no file open to run"));
        return;
    };

    if script.trim().is_empty() {
        state.push_log(LogEntry::new(LogCategory::Err, "empty file — nothing to run"));
        return;
    }

    state.run_output.clear();
    state.run_output_scroll = 0;
    state.run_status = RunStatus::Compiling;
    state.run_start_time = Some(std::time::Instant::now());
    state.run_elapsed_ms = 0;
    state.run_cancel_flag.store(false, Ordering::Relaxed);
    state.show_run_output = true;

    let file_path = state.open_file.clone();
    state.run_target_file = file_path.clone();
    state.last_run_file = file_path;

    state.push_log(LogEntry::new(LogCategory::Jit, "compiling..."));
    state.run_output.push_back(RunOutputLine::new(
        LogCategory::Jit,
        "compiling...",
    ));

    let runner = Arc::clone(runner);
    let tx = tx.clone();
    let cancel = Arc::clone(&state.run_cancel_flag);
    let timeout_secs = state.run_config.timeout_secs;

    std::thread::spawn(move || {
        // Snapshot VRAM before execution for profiling
        let vram_before = {
            let r = runner.lock();
            r.pool_used_bytes()
        };

        let compile_start = std::time::Instant::now();

        // Compile
        let compile_result = {
            let mut r = runner.lock();
            r.compile(&script)
        };

        let compile_ms = compile_start.elapsed().as_millis() as u64;

        match compile_result {
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("compile error: {}", e),
                    is_error: true,
                }).ok();
                tx.send(DaemonEvent::RunFinished {
                    success: false,
                    elapsed_ms: compile_ms,
                }).ok();
                return;
            }
            Ok(()) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("compiled in {}ms", compile_ms),
                    is_error: false,
                }).ok();
            }
        }

        // Check cancel between compile and execute
        if cancel.load(Ordering::Relaxed) {
            tx.send(DaemonEvent::RunOutput {
                line: "cancelled before execution".into(),
                is_error: true,
            }).ok();
            tx.send(DaemonEvent::RunFinished {
                success: false,
                elapsed_ms: compile_ms,
            }).ok();
            return;
        }

        tx.send(DaemonEvent::RunStarted {
            file: String::new(),
        }).ok();

        // Grab program info before execution for profiling
        let program_info = {
            let r = runner.lock();
            r.inspect_last()
        };

        // Execute
        let exec_start = std::time::Instant::now();

        // Timeout watchdog — tracks whether the timeout actually fired
        let timed_out = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancel_timeout = Arc::clone(&cancel);
        let timed_out_flag = Arc::clone(&timed_out);
        let tx_timeout = tx.clone();
        let timeout_handle = std::thread::spawn(move || {
            let deadline = std::time::Duration::from_secs(timeout_secs);
            let start = std::time::Instant::now();
            while start.elapsed() < deadline {
                if cancel_timeout.load(Ordering::Relaxed) {
                    return; // Cancelled before timeout — not a timeout
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            // Timeout fired
            timed_out_flag.store(true, Ordering::Relaxed);
            cancel_timeout.store(true, Ordering::Relaxed);
            let elapsed = start.elapsed().as_millis() as u64;
            tx_timeout.send(DaemonEvent::RunOutput {
                line: format!("TIMEOUT after {}s — execution killed", timeout_secs),
                is_error: true,
            }).ok();
            tx_timeout.send(DaemonEvent::RunTimeout {
                elapsed_ms: elapsed,
            }).ok();
        });

        let exec_result = {
            let r = runner.lock();
            r.execute_last()
        };

        // Snapshot VRAM after execution for profiling delta
        let vram_after = {
            let r = runner.lock();
            r.pool_used_bytes()
        };

        cancel.store(true, Ordering::Relaxed); // Signal timeout thread to stop
        let _ = timeout_handle.join();

        let exec_ms = exec_start.elapsed().as_millis() as u64;
        let total_ms = compile_ms + exec_ms;

        let was_timeout = timed_out.load(Ordering::Relaxed);
        let success = exec_result.is_ok() && !was_timeout;

        match exec_result {
            Ok(result) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!(
                        "executed in {}ms — output shape {:?}, {} values",
                        exec_ms,
                        result.shape,
                        result.data.len()
                    ),
                    is_error: false,
                }).ok();

                // Send first few output values
                let preview: Vec<String> = result.data.iter().take(16).map(|v| format!("{:.4}", v)).collect();
                tx.send(DaemonEvent::RunOutput {
                    line: format!("output: [{}{}]",
                        preview.join(", "),
                        if result.data.len() > 16 { ", ..." } else { "" }
                    ),
                    is_error: false,
                }).ok();

                // Also send tensor result for visualization
                tx.send(DaemonEvent::TensorResult {
                    shape: result.shape,
                    data: result.data,
                }).ok();

                tx.send(DaemonEvent::RunFinished {
                    success: true,
                    elapsed_ms: total_ms,
                }).ok();
            }
            Err(e) => {
                tx.send(DaemonEvent::RunOutput {
                    line: format!("execution error: {}", e),
                    is_error: true,
                }).ok();
                tx.send(DaemonEvent::RunFinished {
                    success: false,
                    elapsed_ms: total_ms,
                }).ok();
            }
        }

        // Emit profiling event
        if let Some(info) = program_info {
            let tag = script.chars().take(40).collect::<String>();
            tx.send(DaemonEvent::KernelProfiled {
                compile_ms,
                exec_ms,
                input_shapes: info.input_shapes,
                output_shape: info.output_shape,
                node_count: info.node_count,
                total_elements: info.total_elements,
                success,
                source_tag: tag,
                vram_before,
                vram_after,
            }).ok();
        }
    });
}

pub fn cmd_stop_run(state: &mut TuiState) {
    if state.run_status == RunStatus::Idle {
        state.push_log(LogEntry::new(LogCategory::Sys, "nothing running"));
        return;
    }
    state.run_cancel_flag.store(true, Ordering::Relaxed);
    state.push_log(LogEntry::new(LogCategory::Sys, "stop signal sent"));
}

pub fn cmd_rerun(
    state: &mut TuiState,
    runner: &Arc<parking_lot::Mutex<ScriptRunner>>,
    tx: &Sender<DaemonEvent>,
) {
    if let Some(path) = state.last_run_file.clone() {
        if state.open_file.as_ref() != Some(&path) {
            match state.open_file_path(path) {
                Ok(()) => {}
                Err(e) => {
                    state.push_log(LogEntry::new(LogCategory::Err, e));
                    return;
                }
            }
        }
    }
    cmd_run(state, runner, tx);
}

pub fn cmd_args(state: &mut TuiState, args: &[&str]) {
    state.run_config.args = args.iter().map(|s| s.to_string()).collect();
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        format!("run args: {:?}", state.run_config.args),
    ));
}

pub fn cmd_profile(state: &mut TuiState, args: &[&str]) {
    match args.first().copied() {
        Some("debug") => {
            state.run_config.profile = RunProfile::Debug;
            state.push_log(LogEntry::new(LogCategory::Sys, "profile: debug"));
        }
        Some("release") => {
            state.run_config.profile = RunProfile::Release;
            state.push_log(LogEntry::new(LogCategory::Sys, "profile: release"));
        }
        _ => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!(
                    "profile: {:?} — usage: profile [debug|release]",
                    state.run_config.profile
                ),
            ));
        }
    }
}

pub fn cmd_timeout(state: &mut TuiState, args: &[&str]) {
    match args.first().copied() {
        Some(val) => {
            match val.parse::<u64>() {
                Ok(secs) if secs > 0 => {
                    state.run_config.timeout_secs = secs;
                    state.push_log(LogEntry::new(
                        LogCategory::Sys,
                        format!("execution timeout: {}s", secs),
                    ));
                }
                _ => {
                    state.push_log(LogEntry::new(
                        LogCategory::Err,
                        "usage: timeout <seconds> (must be positive integer)",
                    ));
                }
            }
        }
        None => {
            state.push_log(LogEntry::new(
                LogCategory::Sys,
                format!("execution timeout: {}s — usage: timeout <seconds>", state.run_config.timeout_secs),
            ));
        }
    }
}
