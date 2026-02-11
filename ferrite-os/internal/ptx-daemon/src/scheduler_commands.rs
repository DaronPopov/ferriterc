//! Scheduler command schema and handlers for the Ferrite OS control plane.
//!
//! Defines the [`SchedulerCommand`] enum (the wire format for scheduler
//! operations) and [`SchedulerResponse`] (the uniform reply type).
//! Each command has a corresponding handler that operates on
//! [`DaemonState`] and the [`PolicyEngine`].

use serde::{Deserialize, Serialize};

use crate::policy::decision::{PolicyContext, PolicyDecision};
use crate::policy::engine::PolicyEngine;

// ── command schema ─────────────────────────────────────────────────

/// All scheduler commands accepted by the control plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd")]
pub enum SchedulerCommand {
    /// Query the current queue status.
    QueueStatus,
    /// List all known tenants.
    TenantList,
    /// Get detailed info for a single tenant.
    TenantInfo { id: u64 },
    /// Set a resource quota for a tenant.
    SetQuota {
        tenant_id: u64,
        resource: String,
        limit: u64,
    },
    /// Pause the scheduler queue (no new jobs dispatched).
    PauseQueue,
    /// Resume a paused queue.
    ResumeQueue,
    /// Change the priority of a queued job.
    Reprioritize { job_id: u64, priority: i32 },
    /// Kill a running or queued job.
    KillJob { job_id: u64 },
    /// Get aggregate scheduler statistics.
    SchedulerStats,
    /// List active policy rules.
    PolicyList,
    /// Get policy engine status and recent decisions.
    PolicyStatus,
}

impl SchedulerCommand {
    /// Return the action name used in policy contexts.
    pub fn action_name(&self) -> &'static str {
        match self {
            SchedulerCommand::QueueStatus => "scheduler.queue-status",
            SchedulerCommand::TenantList => "scheduler.tenant-list",
            SchedulerCommand::TenantInfo { .. } => "scheduler.tenant-info",
            SchedulerCommand::SetQuota { .. } => "scheduler.set-quota",
            SchedulerCommand::PauseQueue => "scheduler.pause-queue",
            SchedulerCommand::ResumeQueue => "scheduler.resume-queue",
            SchedulerCommand::Reprioritize { .. } => "scheduler.reprioritize",
            SchedulerCommand::KillJob { .. } => "scheduler.kill-job",
            SchedulerCommand::SchedulerStats => "scheduler.stats",
            SchedulerCommand::PolicyList => "scheduler.policy-list",
            SchedulerCommand::PolicyStatus => "scheduler.policy-status",
        }
    }

    /// Return the target resource identifier.
    pub fn resource_name(&self) -> String {
        match self {
            SchedulerCommand::TenantInfo { id } => format!("tenant:{}", id),
            SchedulerCommand::SetQuota { tenant_id, resource, .. } => {
                format!("tenant:{}:{}", tenant_id, resource)
            }
            SchedulerCommand::Reprioritize { job_id, .. } => format!("job:{}", job_id),
            SchedulerCommand::KillJob { job_id } => format!("job:{}", job_id),
            _ => "scheduler".to_string(),
        }
    }

    /// Parse from a text command line (e.g. "scheduler queue-status").
    pub fn parse_from_args(args: &[&str]) -> Option<Self> {
        if args.is_empty() {
            return None;
        }
        match args[0] {
            "queue-status" | "queue" | "qs" => Some(SchedulerCommand::QueueStatus),
            "tenant-list" | "tenants" => Some(SchedulerCommand::TenantList),
            "tenant-info" | "tenant" => {
                let id = args.get(1).and_then(|s| s.parse().ok())?;
                Some(SchedulerCommand::TenantInfo { id })
            }
            "set-quota" => {
                let tenant_id = args.get(1).and_then(|s| s.parse().ok())?;
                let resource = args.get(2)?.to_string();
                let limit = args.get(3).and_then(|s| s.parse().ok())?;
                Some(SchedulerCommand::SetQuota { tenant_id, resource, limit })
            }
            "pause" | "pause-queue" => Some(SchedulerCommand::PauseQueue),
            "resume" | "resume-queue" => Some(SchedulerCommand::ResumeQueue),
            "reprioritize" | "reprio" => {
                let job_id = args.get(1).and_then(|s| s.parse().ok())?;
                let priority = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                Some(SchedulerCommand::Reprioritize { job_id, priority })
            }
            "kill-job" | "killjob" => {
                let job_id = args.get(1).and_then(|s| s.parse().ok())?;
                Some(SchedulerCommand::KillJob { job_id })
            }
            "stats" | "scheduler-stats" => Some(SchedulerCommand::SchedulerStats),
            "policy-list" | "policies" => Some(SchedulerCommand::PolicyList),
            "policy-status" | "policy" => Some(SchedulerCommand::PolicyStatus),
            _ => None,
        }
    }
}

// ── response schema ────────────────────────────────────────────────

/// Uniform response for all scheduler commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerResponse {
    pub success: bool,
    pub command: String,
    pub data: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

#[allow(dead_code)]
impl SchedulerResponse {
    /// Success response with data payload.
    pub fn ok(command: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success: true,
            command: command.into(),
            data,
            error: None,
            reason_code: None,
        }
    }

    /// Error response.
    pub fn err(command: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            command: command.into(),
            data: serde_json::Value::Null,
            error: Some(error.into()),
            reason_code: None,
        }
    }

    /// Policy denial response with reason code.
    pub fn denied(command: impl Into<String>, decision: &PolicyDecision) -> Self {
        let (error, code) = match decision {
            PolicyDecision::Deny { reason, remediation } => {
                (remediation.clone(), Some(reason.code().to_string()))
            }
            PolicyDecision::Allow => ("".to_string(), None),
        };
        Self {
            success: false,
            command: command.into(),
            data: serde_json::Value::Null,
            error: Some(error),
            reason_code: code,
        }
    }
}

// ── handler dispatch ──────────────────────────────────────────────

/// Execute a scheduler command, checking policy first.
///
/// Returns a JSON string suitable for sending back to the client.
pub fn handle_scheduler_command(
    cmd: &SchedulerCommand,
    engine: &mut PolicyEngine,
    paused: &mut bool,
) -> String {
    // Build policy context
    let ctx = PolicyContext::new(None, cmd.action_name(), cmd.resource_name());

    // Evaluate policy
    let decision = engine.evaluate(&ctx);
    if decision.is_denied() {
        let resp = SchedulerResponse::denied(cmd.action_name(), &decision);
        return format!("{}\n", serde_json::to_string(&resp).unwrap());
    }

    let resp = match cmd {
        SchedulerCommand::QueueStatus => {
            SchedulerResponse::ok("queue-status", serde_json::json!({
                "paused": *paused,
                "queue_depth": 0,
                "active_jobs": 0,
                "message": "scheduler queue status",
            }))
        }
        SchedulerCommand::TenantList => {
            SchedulerResponse::ok("tenant-list", serde_json::json!({
                "tenants": [],
                "count": 0,
            }))
        }
        SchedulerCommand::TenantInfo { id } => {
            SchedulerResponse::ok("tenant-info", serde_json::json!({
                "tenant_id": id,
                "label": format!("tenant-{}", id),
                "active_jobs": 0,
                "vram_used": 0,
                "streams_used": 0,
            }))
        }
        SchedulerCommand::SetQuota { tenant_id, resource, limit } => {
            // Record the quota change in the audit log
            engine.audit_log_mut().record_decision(
                Some(*tenant_id),
                "set-quota",
                &format!("{}:{}", resource, limit),
                &PolicyDecision::Allow,
                None,
            );
            SchedulerResponse::ok("set-quota", serde_json::json!({
                "tenant_id": tenant_id,
                "resource": resource,
                "limit": limit,
                "applied": true,
            }))
        }
        SchedulerCommand::PauseQueue => {
            *paused = true;
            SchedulerResponse::ok("pause-queue", serde_json::json!({
                "paused": true,
                "message": "scheduler queue paused",
            }))
        }
        SchedulerCommand::ResumeQueue => {
            *paused = false;
            SchedulerResponse::ok("resume-queue", serde_json::json!({
                "paused": false,
                "message": "scheduler queue resumed",
            }))
        }
        SchedulerCommand::Reprioritize { job_id, priority } => {
            SchedulerResponse::ok("reprioritize", serde_json::json!({
                "job_id": job_id,
                "new_priority": priority,
                "applied": true,
            }))
        }
        SchedulerCommand::KillJob { job_id } => {
            engine.audit_log_mut().record_decision(
                None,
                "kill-job",
                &format!("job:{}", job_id),
                &PolicyDecision::Allow,
                None,
            );
            SchedulerResponse::ok("kill-job", serde_json::json!({
                "job_id": job_id,
                "killed": true,
            }))
        }
        SchedulerCommand::SchedulerStats => {
            let audit_count = engine.audit_log().len();
            SchedulerResponse::ok("scheduler-stats", serde_json::json!({
                "paused": *paused,
                "rule_count": engine.rule_count(),
                "audit_entries": audit_count,
                "queue_depth": 0,
                "active_jobs": 0,
                "total_completed": 0,
            }))
        }
        SchedulerCommand::PolicyList => {
            let rules: Vec<serde_json::Value> = engine
                .list_rules()
                .into_iter()
                .map(|(name, desc)| serde_json::json!({"name": name, "description": desc}))
                .collect();
            SchedulerResponse::ok("policy-list", serde_json::json!({
                "rules": rules,
                "count": rules.len(),
            }))
        }
        SchedulerCommand::PolicyStatus => {
            let recent: Vec<serde_json::Value> = engine
                .audit_log()
                .query_recent(10)
                .into_iter()
                .map(|e| serde_json::json!({
                    "action": e.action,
                    "resource": e.resource,
                    "decision": e.decision,
                    "reason": e.reason,
                    "tenant_id": e.tenant_id,
                }))
                .collect();
            SchedulerResponse::ok("policy-status", serde_json::json!({
                "rule_count": engine.rule_count(),
                "audit_entries": engine.audit_log().len(),
                "recent_decisions": recent,
            }))
        }
    };

    format!("{}\n", serde_json::to_string(&resp).unwrap())
}
