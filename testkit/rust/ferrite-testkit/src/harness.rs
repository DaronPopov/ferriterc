use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

use crate::client::parse_json_response;
use crate::env::{default_daemon_config, resolve_daemon_binary};

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub repo_root: PathBuf,
    pub socket_path: Option<PathBuf>,
    pub pid_file: Option<PathBuf>,
    pub headless: bool,
    pub strict_mode: bool,
    pub extra_env: Vec<(String, String)>,
    pub startup_timeout: Duration,
    pub command_timeout: Duration,
}

impl HarnessConfig {
    pub fn new(repo_root: PathBuf) -> Self {
        Self {
            repo_root,
            socket_path: None,
            pid_file: None,
            headless: true,
            strict_mode: false,
            extra_env: Vec::new(),
            startup_timeout: Duration::from_secs(30),
            command_timeout: Duration::from_secs(120),
        }
    }
}

pub struct DaemonHarness {
    child: Option<Child>,
    pub socket_path: PathBuf,
    pub pid_file: PathBuf,
    command_timeout: Duration,
}

impl DaemonHarness {
    pub fn spawn(cfg: HarnessConfig) -> Result<Self> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let test_id = format!("ferrite-testkit-{}-{}", std::process::id(), nonce);

        let socket_path = cfg
            .socket_path
            .unwrap_or_else(|| PathBuf::from(format!("/tmp/{test_id}.sock")));
        let pid_file = cfg
            .pid_file
            .unwrap_or_else(|| PathBuf::from(format!("/tmp/{test_id}.pid")));

        let _ = std::fs::remove_file(&socket_path);
        let _ = std::fs::remove_file(&pid_file);

        let daemon_bin = resolve_daemon_binary(&cfg.repo_root)
            .ok_or_else(|| anyhow!("unable to find daemon binary under ferrite-os/target"))?;

        let mut cmd = Command::new(&daemon_bin);
        cmd.arg("serve");

        let cfg_path = default_daemon_config(&cfg.repo_root);
        if cfg_path.exists() {
            cmd.arg("--config").arg(cfg_path);
        }

        cmd.env("FERRITE_SOCKET", &socket_path)
            .env("FERRITE_PID_FILE", &pid_file)
            .env("FERRITE_HEADLESS", if cfg.headless { "1" } else { "0" })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if cfg.strict_mode {
            cmd.env("FERRITE_SINGLE_POOL_STRICT", "1");
        }

        if let Ok(ld) = std::env::var("LD_LIBRARY_PATH") {
            cmd.env("LD_LIBRARY_PATH", ld);
        }

        for (k, v) in cfg.extra_env {
            cmd.env(k, v);
        }

        let child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn daemon binary {}", daemon_bin.display()))?;

        let mut harness = Self {
            child: Some(child),
            socket_path,
            pid_file,
            command_timeout: cfg.command_timeout,
        };

        harness.wait_ready(cfg.startup_timeout)?;
        Ok(harness)
    }

    fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(ref mut child) = self.child {
                if let Some(status) = child.try_wait().context("failed to poll daemon")? {
                    let diag = self.collect_output();
                    return Err(anyhow!("daemon exited early: {status}\n{diag}"));
                }
            }

            if self.socket_path.exists() {
                if let Ok(raw) = self.try_send_raw("ping", Duration::from_secs(5)) {
                    if parse_json_response(&raw)
                        .ok()
                        .and_then(|v| v.get("message").and_then(|s| s.as_str()).map(|s| s == "pong"))
                        == Some(true)
                    {
                        return Ok(());
                    }
                }
            }

            thread::sleep(Duration::from_millis(200));
        }

        Err(anyhow!(
            "daemon did not become ready within {}s",
            timeout.as_secs()
        ))
    }

    pub fn send_raw(&self, cmd: &str) -> Result<String> {
        self.try_send_raw(cmd, self.command_timeout)
    }

    pub fn send_json(&self, cmd: &str) -> Result<serde_json::Value> {
        let raw = self.send_raw(cmd)?;
        parse_json_response(&raw)
    }

    fn try_send_raw(&self, cmd: &str, timeout: Duration) -> Result<String> {
        let stream = UnixStream::connect(&self.socket_path)
            .with_context(|| format!("connect socket {}", self.socket_path.display()))?;
        stream
            .set_read_timeout(Some(timeout))
            .context("set read timeout")?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .context("set write timeout")?;

        let mut stream = stream;
        stream.write_all(cmd.as_bytes()).context("write command")?;
        stream.write_all(b"\n").context("write newline")?;
        stream.flush().context("flush")?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .context("shutdown write")?;

        let mut out = String::new();
        stream.read_to_string(&mut out).context("read response")?;
        Ok(out)
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.child.is_none() {
            return Ok(());
        }

        let _ = self.try_send_raw("shutdown", Duration::from_secs(5));

        if let Some(ref mut child) = self.child {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(_status) = child.try_wait().context("poll daemon on shutdown")? {
                    break;
                }
                if Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }

        self.child = None;
        let _ = std::fs::remove_file(&self.socket_path);
        let _ = std::fs::remove_file(&self.pid_file);
        Ok(())
    }

    fn collect_output(&mut self) -> String {
        let mut diag = String::new();
        if let Some(ref mut child) = self.child {
            if let Some(ref mut stdout) = child.stdout {
                let mut buf = String::new();
                let _ = stdout.read_to_string(&mut buf);
                if !buf.is_empty() {
                    diag.push_str("=== daemon stdout ===\n");
                    diag.push_str(&buf);
                    diag.push('\n');
                }
            }
            if let Some(ref mut stderr) = child.stderr {
                let mut buf = String::new();
                let _ = stderr.read_to_string(&mut buf);
                if !buf.is_empty() {
                    diag.push_str("=== daemon stderr ===\n");
                    diag.push_str(&buf);
                    diag.push('\n');
                }
            }
        }
        diag
    }
}

impl Drop for DaemonHarness {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[allow(dead_code)]
fn _exists(path: &Path) -> bool {
    path.exists()
}
