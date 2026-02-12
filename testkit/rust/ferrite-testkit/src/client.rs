use anyhow::{Context, Result};
use serde_json::Value;

pub fn parse_json_response(raw: &str) -> Result<Value> {
    let trimmed = raw.trim();
    serde_json::from_str(trimmed)
        .with_context(|| format!("failed to parse daemon JSON response: {trimmed}"))
}

pub fn extract_ok(json: &Value) -> Option<bool> {
    json.get("ok").and_then(|v| v.as_bool())
}

pub fn get_field<'a>(json: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = json;
    for key in path.split('.') {
        cur = cur.get(key)?;
    }
    Some(cur)
}
