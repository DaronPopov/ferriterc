use std::fs;
use std::io;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;

use ptx_runtime::PtxRuntime;
use tracing::{error, info};

use crate::config::DaemonConfig;
use crate::job_store::JobStore;
use crate::pid::PidFile;
use crate::script_runner::ScriptRunner;
use crate::state::DaemonState;
use crate::supervisor::JobSupervisor;

pub(super) fn should_use_tui(config: &DaemonConfig) -> bool {
    !config.headless && unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
}

pub(super) fn init_logging(config: &DaemonConfig, use_tui: bool) {
    if use_tui {
        let log_dir = config.log_dir.clone().unwrap_or_else(|| "/tmp".to_string());
        let file_appender = tracing_appender::rolling::daily(&log_dir, "ferrite-daemon.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        tracing_subscriber::fmt()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .init();
    } else if let Some(log_dir) = &config.log_dir {
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
}

pub(super) fn create_pid_file(config: &DaemonConfig) -> io::Result<PidFile> {
    PidFile::create(Path::new(&config.pid_file)).map_err(|e| {
        error!("Failed to create PID file: {}", e);
        e
    })
}

pub(super) fn prepare_listener(config: &DaemonConfig) -> io::Result<UnixListener> {
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

    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }

    UnixListener::bind(&config.socket_path)
}

pub(super) fn init_runtime(config: &DaemonConfig) -> io::Result<Arc<PtxRuntime>> {
    info!("Initializing GPU runtime on device {}", config.device_id);

    let mut runtime_config = ptx_sys::GPUHotConfig::default();
    runtime_config.max_streams = config.max_streams;
    runtime_config.pool_fraction = config.pool_fraction;
    runtime_config.enable_leak_detection = config.enable_leak_detection;

    let runtime = Arc::new(
        PtxRuntime::with_config(config.device_id, Some(runtime_config)).map_err(|e| {
            error!("Failed to initialize runtime: {:?}", e);
            io::Error::new(
                io::ErrorKind::Other,
                format!("Runtime init failed: {:?}", e),
            )
        })?,
    );

    info!(
        "Runtime initialized with {} streams, pool={:.0}%",
        config.max_streams,
        config.pool_fraction * 100.0
    );

    Ok(runtime)
}

pub(super) fn boot_kernel_if_requested(config: &DaemonConfig, runtime: &Arc<PtxRuntime>) {
    if config.boot_kernel {
        info!("Booting persistent kernel");
        unsafe { ptx_sys::ptx_os_boot_persistent_kernel(runtime.raw()) };
    }
}

pub(super) fn build_state(
    runtime: Arc<PtxRuntime>,
    config: DaemonConfig,
) -> (Arc<DaemonState>, Arc<parking_lot::Mutex<ScriptRunner>>) {
    // Initialize the durable job supervisor.
    let job_store = match JobStore::new(&config.jobs.state_dir) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("failed to create job store at {}: {}; using /tmp fallback", config.jobs.state_dir, e);
            JobStore::new("/tmp/ferrite-jobs-fallback")
                .expect("failed to create fallback job store")
        }
    };

    let (supervisor, reconcile_diags) = match JobSupervisor::new(job_store) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!("job supervisor init failed: {}; starting with empty supervisor", e);
            let fallback_store = JobStore::new("/tmp/ferrite-jobs-fallback")
                .expect("failed to create fallback job store");
            // Create an empty supervisor with no recovered jobs.
            (crate::supervisor::JobSupervisor::empty(fallback_store), Vec::new())
        }
    };

    for diag in &reconcile_diags {
        ptx_runtime::telemetry::emit_diag(diag);
    }

    let state = Arc::new(DaemonState::new(runtime.clone(), config, supervisor));
    let mut script_runner = ScriptRunner::new(runtime);

    // Enable AOT disk cache under the working directory
    let cache_dir = std::env::current_dir()
        .unwrap_or_else(|_| ".".into())
        .join(".ferrite-cache")
        .join("jit");
    if let Err(e) = script_runner.enable_disk_cache(cache_dir) {
        tracing::warn!("failed to enable JIT disk cache: {}", e);
    }

    let runner = Arc::new(parking_lot::Mutex::new(script_runner));
    (state, runner)
}
