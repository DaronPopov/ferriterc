use crate::client::{get_field, parse_json_response};
use crate::harness::DaemonHarness;
use crate::report::{InvariantResult, StepResult};
use crate::scenario::InvariantSpec;

pub fn evaluate_invariants(
    harness: &DaemonHarness,
    steps: &[StepResult],
    specs: &[InvariantSpec],
) -> Vec<InvariantResult> {
    specs
        .iter()
        .map(|spec| evaluate_one(harness, steps, spec))
        .collect()
}

fn evaluate_one(harness: &DaemonHarness, steps: &[StepResult], spec: &InvariantSpec) -> InvariantResult {
    match spec.kind.as_str() {
        "daemon_healthy" => {
            match harness.send_json("status") {
                Ok(v) => {
                    let healthy = v.get("healthy").and_then(|x| x.as_bool()) == Some(true);
                    InvariantResult {
                        kind: spec.kind.clone(),
                        passed: healthy,
                        note: format!("status.healthy={}", v.get("healthy").cloned().unwrap_or_default()),
                    }
                }
                Err(e) => InvariantResult {
                    kind: spec.kind.clone(),
                    passed: false,
                    note: format!("failed to query status: {e}"),
                },
            }
        }
        "json_field_eq" => {
            let Some(command) = spec.command.as_ref() else {
                return fail(spec, "missing command");
            };
            let Some(field) = spec.field.as_ref() else {
                return fail(spec, "missing field");
            };
            let Some(expected) = spec.value.as_ref() else {
                return fail(spec, "missing value");
            };

            match harness.send_json(command) {
                Ok(v) => {
                    let actual = get_field(&v, field);
                    let passed = actual == Some(expected);
                    InvariantResult {
                        kind: spec.kind.clone(),
                        passed,
                        note: format!("field={field} expected={expected} actual={}", actual.cloned().unwrap_or_default()),
                    }
                }
                Err(e) => fail(spec, format!("command failed: {e}")),
            }
        }
        "json_field_contains" => {
            let Some(command) = spec.command.as_ref() else {
                return fail(spec, "missing command");
            };
            let Some(field) = spec.field.as_ref() else {
                return fail(spec, "missing field");
            };
            let Some(substring) = spec.substring.as_ref() else {
                return fail(spec, "missing substring");
            };

            match harness.send_json(command) {
                Ok(v) => {
                    let text = get_field(&v, field).and_then(|f| f.as_str()).unwrap_or("");
                    let passed = text.contains(substring);
                    InvariantResult {
                        kind: spec.kind.clone(),
                        passed,
                        note: format!("field={field} contains={substring} actual={text}"),
                    }
                }
                Err(e) => fail(spec, format!("command failed: {e}")),
            }
        }
        "stdout_contains" => step_text_check(steps, spec, true, false),
        "stderr_contains" => step_text_check(steps, spec, false, false),
        "stderr_not_contains" => step_text_check(steps, spec, false, true),
        _ => fail(spec, format!("unsupported invariant kind '{}'", spec.kind)),
    }
}

fn step_text_check(steps: &[StepResult], spec: &InvariantSpec, stdout: bool, negate: bool) -> InvariantResult {
    let Some(step_index) = spec.step_index else {
        return fail(spec, "missing step_index");
    };
    let Some(substring) = spec.substring.as_ref() else {
        return fail(spec, "missing substring");
    };
    let Some(step) = steps.get(step_index) else {
        return fail(spec, format!("step_index out of range: {step_index}"));
    };

    let parsed = parse_json_response(&step.raw).ok();
    let text = if stdout {
        parsed
            .as_ref()
            .and_then(|v| v.get("stdout"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
    } else {
        parsed
            .as_ref()
            .and_then(|v| v.get("stderr"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
    };

    let contains = text.contains(substring);
    let passed = if negate { !contains } else { contains };

    InvariantResult {
        kind: spec.kind.clone(),
        passed,
        note: format!(
            "step={} field={} substring={} contains={}",
            step_index,
            if stdout { "stdout" } else { "stderr" },
            substring,
            contains
        ),
    }
}

fn fail(spec: &InvariantSpec, note: impl Into<String>) -> InvariantResult {
    InvariantResult {
        kind: spec.kind.clone(),
        passed: false,
        note: note.into(),
    }
}
