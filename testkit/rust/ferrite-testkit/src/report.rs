use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub kind: String,
    pub send: Option<String>,
    pub raw: String,
    pub ok: Option<bool>,
    pub passed: bool,
    pub note: Option<String>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantResult {
    pub kind: String,
    pub passed: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub name: String,
    pub description: Option<String>,
    pub passed: bool,
    pub started_at_epoch_ms: u128,
    pub elapsed_ms: u128,
    pub steps: Vec<StepResult>,
    pub invariants: Vec<InvariantResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub scenarios: Vec<ScenarioResult>,
}

pub fn scenario_error(name: impl Into<String>, detail: impl Into<String>) -> ScenarioResult {
    ScenarioResult {
        name: name.into(),
        description: Some("scenario execution failed".to_string()),
        passed: false,
        started_at_epoch_ms: now_epoch_ms(),
        elapsed_ms: 0,
        steps: Vec::new(),
        invariants: vec![InvariantResult {
            kind: "scenario_error".to_string(),
            passed: false,
            note: detail.into(),
        }],
    }
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create report dir {}", parent.display()))?;
    }
    fs::write(path, json).with_context(|| format!("failed to write report {}", path.display()))
}

pub fn now_epoch_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}
