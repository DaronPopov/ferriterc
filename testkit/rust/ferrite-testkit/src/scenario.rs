use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::client::{extract_ok, parse_json_response};
use crate::harness::{DaemonHarness, HarnessConfig};
use crate::invariants::evaluate_invariants;
use crate::report::{now_epoch_ms, InvariantResult, ScenarioResult, StepResult};

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub strict_override: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioSpec {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub daemon: DaemonProfile,
    #[serde(default)]
    pub steps: Vec<StepSpec>,
    #[serde(default)]
    pub invariants: Vec<InvariantSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DaemonProfile {
    #[serde(default)]
    pub strict_mode: bool,
    #[serde(default = "default_headless")]
    pub headless: bool,
    pub command_timeout_ms: Option<u64>,
    pub startup_timeout_ms: Option<u64>,
    pub env: Option<BTreeMap<String, String>>,
}

impl Default for DaemonProfile {
    fn default() -> Self {
        Self {
            strict_mode: false,
            headless: true,
            command_timeout_ms: None,
            startup_timeout_ms: None,
            env: None,
        }
    }
}

fn default_headless() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StepSpec {
    Command {
        send: String,
        expect_ok: Option<bool>,
    },
    Sleep {
        sleep_ms: u64,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct InvariantSpec {
    pub kind: String,
    pub command: Option<String>,
    pub field: Option<String>,
    pub value: Option<Value>,
    pub substring: Option<String>,
    pub step_index: Option<usize>,
}

pub fn run_daemon_smoke(repo_root: &Path) -> Result<ScenarioResult> {
    let spec = ScenarioSpec {
        name: "daemon-smoke".to_string(),
        description: Some("simple daemon ping/status/health smoke".to_string()),
        daemon: DaemonProfile::default(),
        steps: vec![
            StepSpec::Command {
                send: "ping".to_string(),
                expect_ok: Some(true),
            },
            StepSpec::Command {
                send: "status".to_string(),
                expect_ok: Some(true),
            },
            StepSpec::Command {
                send: "health".to_string(),
                expect_ok: Some(true),
            },
        ],
        invariants: vec![InvariantSpec {
            kind: "daemon_healthy".to_string(),
            command: None,
            field: None,
            value: None,
            substring: None,
            step_index: None,
        }],
    };

    run_scenario_spec(&spec, repo_root, &RunOptions::default())
}

pub fn run_scenario_file(path: &Path, repo_root: &Path, options: &RunOptions) -> Result<ScenarioResult> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read scenario {}", path.display()))?;
    let spec: ScenarioSpec = toml::from_str(&raw)
        .with_context(|| format!("failed to parse scenario TOML {}", path.display()))?;

    run_scenario_spec(&spec, repo_root, options)
}

fn run_scenario_spec(spec: &ScenarioSpec, repo_root: &Path, options: &RunOptions) -> Result<ScenarioResult> {
    let started_at = now_epoch_ms();
    let start = Instant::now();

    let mut cfg = HarnessConfig::new(repo_root.to_path_buf());
    cfg.headless = spec.daemon.headless;
    cfg.strict_mode = options.strict_override.unwrap_or(spec.daemon.strict_mode);
    if let Some(ms) = spec.daemon.command_timeout_ms {
        cfg.command_timeout = Duration::from_millis(ms);
    }
    if let Some(ms) = spec.daemon.startup_timeout_ms {
        cfg.startup_timeout = Duration::from_millis(ms);
    }
    if let Some(env) = &spec.daemon.env {
        cfg.extra_env = env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    }

    let mut harness = DaemonHarness::spawn(cfg)?;

    let mut steps: Vec<StepResult> = Vec::new();

    for step in &spec.steps {
        match step {
            StepSpec::Sleep { sleep_ms } => {
                let step_start = Instant::now();
                thread::sleep(Duration::from_millis(*sleep_ms));
                steps.push(StepResult {
                    kind: "sleep".to_string(),
                    send: None,
                    raw: String::new(),
                    ok: Some(true),
                    passed: true,
                    note: Some(format!("slept {sleep_ms}ms")),
                    elapsed_ms: step_start.elapsed().as_millis(),
                });
            }
            StepSpec::Command { send, expect_ok } => {
                let step_start = Instant::now();
                let raw = match harness.send_raw(send) {
                    Ok(r) => r,
                    Err(e) => {
                        steps.push(StepResult {
                            kind: "command".to_string(),
                            send: Some(send.clone()),
                            raw: String::new(),
                            ok: None,
                            passed: false,
                            note: Some(format!("command failed: {e}")),
                            elapsed_ms: step_start.elapsed().as_millis(),
                        });
                        break;
                    }
                };

                let json = parse_json_response(&raw).ok();
                let ok = json.as_ref().and_then(extract_ok);
                let passed = expect_ok.map(|exp| ok == Some(exp)).unwrap_or(true);

                steps.push(StepResult {
                    kind: "command".to_string(),
                    send: Some(send.clone()),
                    raw,
                    ok,
                    passed,
                    note: expect_ok.map(|exp| format!("expected ok={exp}")),
                    elapsed_ms: step_start.elapsed().as_millis(),
                });
            }
        }
    }

    let invariants: Vec<InvariantResult> = evaluate_invariants(&harness, &steps, &spec.invariants);

    harness.shutdown().ok();

    let passed_steps = steps.iter().all(|s| s.passed);
    let passed_invariants = invariants.iter().all(|i| i.passed);

    Ok(ScenarioResult {
        name: spec.name.clone(),
        description: spec.description.clone(),
        passed: passed_steps && passed_invariants,
        started_at_epoch_ms: started_at,
        elapsed_ms: start.elapsed().as_millis(),
        steps,
        invariants,
    })
}

#[allow(dead_code)]
fn _ensure_non_empty(spec: &ScenarioSpec) -> Result<()> {
    if spec.name.trim().is_empty() {
        return Err(anyhow!("scenario name must not be empty"));
    }
    Ok(())
}
