use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use tracing::{debug, error, trace, warn};

use crate::config::DaemonConfig;
use crate::events::DaemonEvent;
use crate::script_runner::ScriptRunner;
use crate::state::{ClientGuard, DaemonState};

use super::command_pipeline;

pub(super) fn run_socket_listener(
    listener: UnixListener,
    state: Arc<DaemonState>,
    runner: Arc<parking_lot::Mutex<ScriptRunner>>,
    config: DaemonConfig,
    event_tx: Option<mpsc::Sender<DaemonEvent>>,
) {
    listener
        .set_nonblocking(true)
        .expect("Cannot set non-blocking");

    while state.is_running() {
        match listener.accept() {
            Ok((stream, _)) => {
                let active = state.active_clients.load(Ordering::Relaxed);
                if active >= config.max_clients as u64 {
                    warn!(
                        "Max clients reached ({}), rejecting connection",
                        config.max_clients
                    );
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    continue;
                }

                let state_clone = Arc::clone(&state);
                let runner_clone = Arc::clone(&runner);
                let tx_clone = event_tx.clone();
                thread::spawn(move || {
                    if let Err(e) = handle_client(stream, state_clone, runner_clone, tx_clone) {
                        debug!("Client error: {}", e);
                    }
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                error!("Accept error: {}", e);
            }
        }
    }
}

fn handle_client(
    mut stream: UnixStream,
    state: Arc<DaemonState>,
    _runner: Arc<parking_lot::Mutex<ScriptRunner>>,
    event_tx: Option<mpsc::Sender<DaemonEvent>>,
) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(state.config.client_timeout_secs)))?;
    stream.set_nonblocking(false)?;

    state.active_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard::new(&state);

    let mut buf = String::new();
    stream.read_to_string(&mut buf)?;
    let cmdline = buf.trim();

    let Some(parsed) = command_pipeline::parse_command_line(cmdline) else {
        stream.write_all(b"{\"error\":\"empty command\"}\n")?;
        state.failed_requests.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    };

    trace!(command = %parsed.command, args = ?parsed.args, "Handling client request");
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Agent commands are routed through the TUI event channel for
    // synchronous execution on the main thread.
    if parsed.command == "agent" {
        if let Some(ref tx) = event_tx {
            let json_str = parsed.args.join(" ");
            match serde_json::from_str::<crate::tui::agent::protocol::AgentCommand>(&json_str) {
                Ok(cmd) => {
                    let (reply_tx, reply_rx) = mpsc::channel();
                    tx.send(DaemonEvent::AgentCommandSync(cmd, reply_tx)).ok();
                    match reply_rx.recv_timeout(Duration::from_secs(30)) {
                        Ok(resp) => {
                            let json = serde_json::to_string(&resp).unwrap_or_else(|_| {
                                "{\"error\":\"serialize failed\"}".to_string()
                            });
                            stream.write_all(json.as_bytes())?;
                            stream.write_all(b"\n")?;
                        }
                        Err(_) => {
                            stream.write_all(b"{\"error\":\"timeout waiting for TUI\"}\n")?;
                        }
                    }
                }
                Err(e) => {
                    let msg = format!("{{\"error\":\"invalid agent JSON: {}\"}}\n", e);
                    stream.write_all(msg.as_bytes())?;
                }
            }
            if let Some(ref tx2) = event_tx {
                tx2.send(DaemonEvent::ClientHandled {
                    command: parsed.command,
                    success: true,
                })
                .ok();
            }
            return Ok(());
        }
    }

    let response = command_pipeline::execute_command(&state, &parsed);
    let success = response.is_ok();

    match response {
        Ok(msg) => {
            stream.write_all(msg.as_bytes())?;
        }
        Err(e) => {
            let err_msg = format!("{{\"error\":\"{}\"}}\n", e);
            stream.write_all(err_msg.as_bytes())?;
            state.failed_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    if let Some(tx) = event_tx {
        tx.send(DaemonEvent::ClientHandled {
            command: parsed.command,
            success,
        })
        .ok();
    }

    Ok(())
}
