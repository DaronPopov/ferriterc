// Ferrite-OS Daemon - Production-Grade GPU Runtime Daemon
//
// Features:
// - Signal handling (SIGTERM, SIGINT, SIGHUP)
// - Structured logging with tracing
// - Graceful shutdown
// - Concurrent client handling
// - Rate limiting
// - PID file management
// - Health monitoring
// - Configuration file support

use std::env;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};
use ptx_runtime::PtxRuntime;
use serde::{Deserialize, Serialize};
use signal_hook::consts::{SIGTERM, SIGINT, SIGHUP};
use signal_hook::iterator::Signals;
use tracing::{error, info, warn, debug, trace};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DaemonConfig {
    /// GPU device ID
    #[serde(default)]
    device_id: i32,

    /// Unix socket path
    #[serde(default = "default_socket_path")]
    socket_path: String,

    /// PID file path
    #[serde(default = "default_pid_path")]
    pid_file: String,

    /// Maximum concurrent clients
    #[serde(default = "default_max_clients")]
    max_clients: usize,

    /// Maximum streams
    #[serde(default = "default_max_streams")]
    max_streams: u32,

    /// VRAM pool fraction
    #[serde(default = "default_pool_fraction")]
    pool_fraction: f32,

    /// Keepalive interval (milliseconds)
    #[serde(default = "default_keepalive_ms")]
    keepalive_ms: u64,

    /// Watch metrics interval (milliseconds)
    #[serde(default = "default_watch_ms")]
    watch_ms: u64,

    /// Enable watch mode
    #[serde(default)]
    watch_enabled: bool,

    /// Enable leak detection
    #[serde(default)]
    enable_leak_detection: bool,

    /// Boot persistent kernel
    #[serde(default)]
    boot_kernel: bool,

    /// Log directory
    #[serde(default = "default_log_dir")]
    log_dir: Option<String>,

    /// Client timeout (seconds)
    #[serde(default = "default_client_timeout")]
    client_timeout_secs: u64,

    /// Optional directory containing managed app binaries
    #[serde(default = "default_apps_bin_dir")]
    apps_bin_dir: Option<String>,
}

fn default_socket_path() -> String {
    let uid = unsafe { libc::geteuid() };
    format!("/var/run/ferrite-os/daemon_{}.sock", uid)
}

fn default_pid_path() -> String {
    let uid = unsafe { libc::geteuid() };
    format!("/var/run/ferrite-os/daemon_{}.pid", uid)
}

fn default_max_clients() -> usize { 32 }
fn default_max_streams() -> u32 { 16 }
fn default_pool_fraction() -> f32 { 0.6 }
fn default_keepalive_ms() -> u64 { 5000 }
fn default_watch_ms() -> u64 { 1000 }
fn default_log_dir() -> Option<String> { None }
fn default_client_timeout() -> u64 { 30 }
fn default_apps_bin_dir() -> Option<String> { None }

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            socket_path: default_socket_path(),
            pid_file: default_pid_path(),
            max_clients: default_max_clients(),
            max_streams: default_max_streams(),
            pool_fraction: default_pool_fraction(),
            keepalive_ms: default_keepalive_ms(),
            watch_ms: default_watch_ms(),
            watch_enabled: false,
            enable_leak_detection: false,
            boot_kernel: false,
            log_dir: default_log_dir(),
            client_timeout_secs: default_client_timeout(),
            apps_bin_dir: default_apps_bin_dir(),
        }
    }
}

impl DaemonConfig {
    fn load_from_file(path: &Path) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        toml::from_str(&contents).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("TOML parse error: {}", e))
        })
    }

    fn merge_from_env(&mut self) {
        if let Ok(val) = env::var("FERRITE_DEVICE") {
            if let Ok(id) = val.parse() {
                self.device_id = id;
            }
        }
        if let Ok(val) = env::var("FERRITE_SOCKET") {
            self.socket_path = val;
        }
        if let Ok(val) = env::var("FERRITE_MAX_STREAMS") {
            if let Ok(streams) = val.parse() {
                self.max_streams = streams;
            }
        }
        if let Ok(_) = env::var("FERRITE_BOOT_KERNEL") {
            self.boot_kernel = true;
        }
        if let Ok(_) = env::var("FERRITE_WATCH") {
            self.watch_enabled = true;
        }
        if let Ok(val) = env::var("FERRITE_APPS_BIN_DIR") {
            self.apps_bin_dir = Some(val);
        }
    }
}

// ============================================================================
// Daemon State
// ============================================================================

struct DaemonState {
    runtime: Arc<PtxRuntime>,
    config: DaemonConfig,
    start_time: Instant,
    active_clients: AtomicU64,
    total_requests: AtomicU64,
    failed_requests: AtomicU64,
    running: AtomicBool,
    apps: parking_lot::Mutex<HashMap<u64, ManagedApp>>,
    next_app_id: AtomicU64,
}

struct ManagedApp {
    id: u64,
    name: String,
    args: Vec<String>,
    started_at: Instant,
    child: Child,
}

impl DaemonState {
    fn new(runtime: Arc<PtxRuntime>, config: DaemonConfig) -> Self {
        Self {
            runtime,
            config,
            start_time: Instant::now(),
            active_clients: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            running: AtomicBool::new(true),
            apps: parking_lot::Mutex::new(HashMap::new()),
            next_app_id: AtomicU64::new(1),
        }
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

const MANAGED_APPS: &[&str] = &[
    "vram_database",
    "neural_fabric",
    "stream_compute",
    "checkpoint_engine",
    "gpu_nas",
];

// ============================================================================
// PID File Management
// ============================================================================

struct PidFile {
    path: PathBuf,
}

impl PidFile {
    fn create(path: &Path) -> io::Result<Self> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Check if daemon already running
        if path.exists() {
            if let Ok(contents) = fs::read_to_string(path) {
                if let Ok(pid) = contents.trim().parse::<i32>() {
                    // Check if process exists
                    if unsafe { libc::kill(pid, 0) } == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            format!("Daemon already running (PID: {})", pid),
                        ));
                    }
                }
            }
            // Stale PID file, remove it
            fs::remove_file(path)?;
        }

        // Write our PID
        let pid = unsafe { libc::getpid() };
        fs::write(path, format!("{}\n", pid))?;

        Ok(Self {
            path: path.to_path_buf(),
        })
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

// ============================================================================
// Request Handling
// ============================================================================

fn handle_client(
    mut stream: UnixStream,
    state: Arc<DaemonState>,
) -> io::Result<()> {
    // Set read timeout
    stream.set_read_timeout(Some(Duration::from_secs(state.config.client_timeout_secs)))?;

    state.active_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard::new(&state);

    let mut buf = String::new();
    stream.read_to_string(&mut buf)?;
    let cmdline = buf.trim();
    let mut parts = cmdline.split_whitespace();
    let cmd = parts.next().unwrap_or("").to_lowercase();
    let args: Vec<&str> = parts.collect();

    trace!(command = %cmd, args = ?args, "Handling client request");

    if cmd.is_empty() {
        stream.write_all(b"{\"error\":\"empty command\"}\n")?;
        state.failed_requests.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }

    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let response = match cmd.as_str() {
        "ping" => handle_ping(),
        "status" => handle_status(&state),
        "stats" => handle_stats(&state),
        "metrics" => handle_metrics(&state),
        "snapshot" => handle_snapshot(&state),
        "health" => handle_health(&state),
        "keepalive" => handle_keepalive(&state),
        "apps" => handle_apps(&state),
        "app-start" => handle_app_start(&state, &args),
        "app-stop" => handle_app_stop(&state, &args),
        "shutdown" => {
            state.shutdown();
            Ok("{\"ok\":true,\"message\":\"shutting down\"}\n".to_string())
        }
        "help" => handle_help(),
        _ => {
            state.failed_requests.fetch_add(1, Ordering::Relaxed);
            Ok(format!("{{\"error\":\"unknown command\",\"command\":\"{}\"}}\n", cmd))
        }
    };

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

    Ok(())
}

struct ClientGuard<'a> {
    state: &'a DaemonState,
}

impl<'a> ClientGuard<'a> {
    fn new(state: &'a DaemonState) -> Self {
        Self { state }
    }
}

impl<'a> Drop for ClientGuard<'a> {
    fn drop(&mut self) {
        self.state.active_clients.fetch_sub(1, Ordering::Relaxed);
    }
}

// ============================================================================
// Command Handlers
// ============================================================================

fn handle_ping() -> io::Result<String> {
    Ok("{\"ok\":true,\"message\":\"pong\"}\n".to_string())
}

fn cleanup_exited_apps(state: &DaemonState) {
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

fn resolve_app_target(config: &DaemonConfig, app: &str) -> PathBuf {
    if let Some(dir) = &config.apps_bin_dir {
        return Path::new(dir).join(app);
    }
    PathBuf::from(app)
}

fn handle_apps(state: &DaemonState) -> io::Result<String> {
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

fn handle_app_start(state: &DaemonState, args: &[&str]) -> io::Result<String> {
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

fn handle_app_stop(state: &DaemonState, args: &[&str]) -> io::Result<String> {
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

fn handle_status(state: &DaemonState) -> io::Result<String> {
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

fn handle_stats(state: &DaemonState) -> io::Result<String> {
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

fn handle_metrics(state: &DaemonState) -> io::Result<String> {
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

fn handle_snapshot(state: &DaemonState) -> io::Result<String> {
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

fn handle_health(state: &DaemonState) -> io::Result<String> {
    cleanup_exited_apps(state);
    let tlsf = state.runtime.tlsf_stats();
    let uptime = state.start_time.elapsed().as_secs();
    let active = state.active_clients.load(Ordering::Relaxed);
    let running_apps = state.apps.lock().len();

    let healthy = tlsf.is_healthy &&
                  active < state.config.max_clients as u64 &&
                  tlsf.utilization_percent < 95.0;

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

fn handle_keepalive(state: &DaemonState) -> io::Result<String> {
    state.runtime.keepalive();
    Ok("{\"ok\":true,\"message\":\"keepalive sent\"}\n".to_string())
}

fn handle_help() -> io::Result<String> {
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
    "shutdown": "Shutdown daemon",
    "help": "Show this help"
  }
}
"#;
    Ok(help.to_string())
}

// ============================================================================
// Background Tasks
// ============================================================================

fn start_keepalive_thread(state: Arc<DaemonState>) {
    let keepalive_ms = state.config.keepalive_ms;
    thread::spawn(move || {
        info!("Keepalive thread started (interval: {}ms)", keepalive_ms);
        while state.is_running() {
            state.runtime.keepalive();
            thread::sleep(Duration::from_millis(keepalive_ms));
        }
        info!("Keepalive thread stopped");
    });
}

fn start_watch_thread(state: Arc<DaemonState>) {
    if !state.config.watch_enabled {
        return;
    }

    let watch_ms = state.config.watch_ms;
    let is_tty = unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 };

    thread::spawn(move || {
        info!("Watch thread started (interval: {}ms)", watch_ms);
        while state.is_running() {
            let tlsf = state.runtime.tlsf_stats();
            let stats = state.runtime.stats();
            let active = state.active_clients.load(Ordering::Relaxed);

            let line = format!(
                "util={:.1}% frag={:.4}% vram={}MB streams={} clients={}",
                tlsf.utilization_percent,
                tlsf.fragmentation_ratio * 100.0,
                stats.vram_used / (1024 * 1024),
                stats.active_streams,
                active
            );

            if is_tty {
                print!("\r{}", line);
                let _ = io::stdout().flush();
            } else {
                println!("{}", line);
            }

            thread::sleep(Duration::from_millis(watch_ms));
        }
        if is_tty {
            println!();
        }
        info!("Watch thread stopped");
    });
}

fn start_signal_handler(state: Arc<DaemonState>) {
    thread::spawn(move || {
        let mut signals = Signals::new(&[SIGTERM, SIGINT, SIGHUP])
            .expect("Failed to register signal handlers");

        for sig in signals.forever() {
            match sig {
                SIGTERM | SIGINT => {
                    info!("Received signal {}, initiating shutdown", sig);
                    state.shutdown();
                    break;
                }
                SIGHUP => {
                    info!("Received SIGHUP, reloading configuration not yet implemented");
                }
                _ => {}
            }
        }
    });
}

// ============================================================================
// Server
// ============================================================================

fn run_server(config: DaemonConfig) -> io::Result<()> {
    // Initialize logging
    if let Some(log_dir) = &config.log_dir {
        let file_appender = tracing_appender::rolling::daily(log_dir, "ferrite-daemon.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        tracing_subscriber::fmt()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_target(true)
            .with_thread_ids(true)
            .init();
    }

    info!("Starting Ferrite-OS daemon");
    info!("Configuration: {:?}", config);

    // Create PID file
    let _pid_file = PidFile::create(Path::new(&config.pid_file))
        .map_err(|e| {
            error!("Failed to create PID file: {}", e);
            e
        })?;

    info!("PID file created: {}", config.pid_file);

    // Clean up stale socket
    let socket_path = Path::new(&config.socket_path);
    if socket_path.exists() {
        if UnixStream::connect(socket_path).is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "Daemon already running",
            ));
        }
        fs::remove_file(socket_path)?;
    }

    // Create socket directory
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Initialize runtime
    info!("Initializing GPU runtime on device {}", config.device_id);
    let mut runtime_config = ptx_sys::GPUHotConfig::default();
    runtime_config.max_streams = config.max_streams;
    runtime_config.pool_fraction = config.pool_fraction;
    runtime_config.enable_leak_detection = config.enable_leak_detection;

    let runtime = Arc::new(
        PtxRuntime::with_config(config.device_id, Some(runtime_config))
            .map_err(|e| {
                error!("Failed to initialize runtime: {:?}", e);
                io::Error::new(io::ErrorKind::Other, format!("Runtime init failed: {:?}", e))
            })?
    );

    info!(
        "Runtime initialized with {} streams, pool={:.0}%",
        config.max_streams,
        config.pool_fraction * 100.0
    );

    // Boot persistent kernel if requested
    if config.boot_kernel {
        info!("Booting persistent kernel");
        unsafe { ptx_sys::ptx_os_boot_persistent_kernel(runtime.raw()) };
    }

    // Bind socket
    let listener = UnixListener::bind(&config.socket_path)?;
    info!("Daemon listening on {}", config.socket_path);

    // Create daemon state
    let state = Arc::new(DaemonState::new(runtime, config.clone()));

    // Start background threads
    start_signal_handler(Arc::clone(&state));
    start_keepalive_thread(Arc::clone(&state));
    start_watch_thread(Arc::clone(&state));

    info!("Daemon ready");

    // Accept connections
    for stream in listener.incoming() {
        if !state.is_running() {
            break;
        }

        match stream {
            Ok(stream) => {
                // Check client limit
                let active = state.active_clients.load(Ordering::Relaxed);
                if active >= config.max_clients as u64 {
                    warn!("Max clients reached ({}), rejecting connection", config.max_clients);
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    continue;
                }

                // Handle client in new thread
                let state_clone = Arc::clone(&state);
                thread::spawn(move || {
                    if let Err(e) = handle_client(stream, state_clone) {
                        debug!("Client error: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Accept error: {}", e);
            }
        }
    }

    info!("Shutting down daemon");

    // Stop managed apps
    {
        let mut apps = state.apps.lock();
        for (_, app) in apps.iter_mut() {
            if let Err(e) = app.child.kill() {
                warn!(app_id = app.id, app = %app.name, error = %e, "Failed to kill managed app");
                continue;
            }
            if let Err(e) = app.child.wait() {
                warn!(app_id = app.id, app = %app.name, error = %e, "Failed waiting managed app");
            }
        }
        apps.clear();
    }

    // Cleanup
    let _ = fs::remove_file(&config.socket_path);

    info!("Daemon stopped");
    Ok(())
}

// ============================================================================
// Client Commands
// ============================================================================

fn connect_and_send(socket: &str, command: &str) -> io::Result<()> {
    let mut stream = UnixStream::connect(socket)?;
    stream.write_all(command.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;

    let mut resp = String::new();
    stream.read_to_string(&mut resp)?;
    print!("{}", resp);
    Ok(())
}

fn run_watch_client(socket: &str, watch_ms: u64) -> io::Result<()> {
    let is_tty = unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 };

    loop {
        let mut stream = UnixStream::connect(socket)?;
        stream.write_all(b"metrics\n")?;
        stream.flush()?;

        let mut resp = String::new();
        stream.read_to_string(&mut resp)?;

        if is_tty {
            print!("\r{}", resp.trim());
            io::stdout().flush()?;
        } else {
            println!("{}", resp.trim());
        }

        thread::sleep(Duration::from_millis(watch_ms));
    }
}

// ============================================================================
// Main
// ============================================================================

fn print_usage() {
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
    eprintln!();
    eprintln!("ENVIRONMENT:");
    eprintln!("    FERRITE_DEVICE        GPU device ID");
    eprintln!("    FERRITE_SOCKET        Socket path");
    eprintln!("    FERRITE_MAX_STREAMS   Maximum streams");
    eprintln!("    FERRITE_BOOT_KERNEL   Boot persistent kernel");
    eprintln!("    FERRITE_WATCH         Enable watch mode");
    eprintln!("    FERRITE_APPS_BIN_DIR  Managed app binaries directory");
}

fn main() {
    // Ignore SIGPIPE
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }

    let mut config = DaemonConfig::default();
    let mut command: Option<String> = None;
    let mut command_args: Vec<String> = Vec::new();
    let mut args = env::args().skip(1).peekable();

    // Parse command line arguments
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                if let Some(path) = args.next() {
                    match DaemonConfig::load_from_file(Path::new(&path)) {
                        Ok(loaded) => config = loaded,
                        Err(e) => {
                            eprintln!("Error loading config: {}", e);
                            std::process::exit(1);
                        }
                    }
                } else {
                    eprintln!("--config requires a value");
                    std::process::exit(2);
                }
            }
            "--socket" => {
                config.socket_path = args.next().expect("--socket requires a value");
            }
            "--device" => {
                config.device_id = args.next()
                    .expect("--device requires a value")
                    .parse()
                    .expect("Invalid device ID");
            }
            "--streams" => {
                config.max_streams = args.next()
                    .expect("--streams requires a value")
                    .parse()
                    .expect("Invalid streams value");
            }
            "--pool-fraction" => {
                config.pool_fraction = args.next()
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
                config.watch_ms = args.next()
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
            arg if !arg.starts_with("--") => {
                if command.is_none() {
                    command = Some(arg.to_string());
                } else {
                    command_args.push(arg.to_string());
                }
            }
            _ => {
                eprintln!("Unknown option: {}", arg);
                std::process::exit(2);
            }
        }
    }

    // Merge environment variables
    config.merge_from_env();

    let command = command.unwrap_or_else(|| "serve".to_string());

    match command.as_str() {
        "serve" | "start" => {
            if let Err(e) = run_server(config) {
                eprintln!("Daemon error: {}", e);
                std::process::exit(1);
            }
        }
        "watch" => {
            if let Err(e) = run_watch_client(&config.socket_path, config.watch_ms) {
                eprintln!("Watch error: {}", e);
                std::process::exit(1);
            }
        }
        cmd @ ("ping" | "status" | "stats" | "metrics" | "snapshot" | "health" | "keepalive" | "shutdown" | "help" | "apps" | "app-start" | "app-stop") => {
            let mut line = cmd.to_string();
            for arg in &command_args {
                line.push(' ');
                line.push_str(arg);
            }
            if let Err(e) = connect_and_send(&config.socket_path, &line) {
                eprintln!("Command error: {}", e);
                std::process::exit(1);
            }
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}
