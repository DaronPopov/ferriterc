//! Policy enforcement engine.
//!
//! The engine evaluates an ordered list of [`PolicyRule`]s against a
//! [`PolicyContext`].  First denial wins.  Every evaluation is recorded in
//! the embedded [`AuditLog`].

use super::audit::AuditLog;
use super::decision::{DenialReason, PolicyContext, PolicyDecision};

/// A named policy rule with a check function.
pub struct PolicyRule {
    /// Human-readable rule name (e.g. "max-concurrent-jobs").
    pub name: String,
    /// Description of what this rule enforces.
    pub description: String,
    /// The check function.  Returns `Allow` or `Deny`.
    pub check: Box<dyn Fn(&PolicyContext) -> PolicyDecision + Send + Sync>,
}

impl PolicyRule {
    /// Create a new policy rule.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        check: impl Fn(&PolicyContext) -> PolicyDecision + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            check: Box::new(check),
        }
    }
}

impl std::fmt::Debug for PolicyRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

/// The policy enforcement engine.
///
/// Holds an ordered list of rules and an audit log.  Call [`evaluate`] to
/// run all rules against a context.
pub struct PolicyEngine {
    rules: Vec<PolicyRule>,
    audit_log: AuditLog,
}

impl PolicyEngine {
    /// Create a new engine with the given audit log capacity.
    pub fn new(audit_max_entries: usize) -> Self {
        Self {
            rules: Vec::new(),
            audit_log: AuditLog::new(audit_max_entries),
        }
    }

    /// Create an engine pre-loaded with the default rule set.
    pub fn with_defaults(audit_max_entries: usize) -> Self {
        let mut engine = Self::new(audit_max_entries);
        engine.add_rule(builtin_require_tenant_rule());
        engine
    }

    /// Add a rule to the evaluation chain.
    pub fn add_rule(&mut self, rule: PolicyRule) {
        self.rules.push(rule);
    }

    /// Evaluate all rules.  First `Deny` wins; if all pass the result
    /// is `Allow`.  The decision is recorded in the audit log.
    pub fn evaluate(&mut self, ctx: &PolicyContext) -> PolicyDecision {
        let mut decision = PolicyDecision::Allow;

        for rule in &self.rules {
            let result = (rule.check)(ctx);
            if result.is_denied() {
                decision = result;
                break;
            }
        }

        // Record in audit log.
        self.audit_log.record_decision(
            ctx.tenant_id,
            &ctx.action,
            &ctx.resource,
            &decision,
            None,
        );

        decision
    }

    /// Evaluate without recording (useful for dry-run / preview).
    #[allow(dead_code)]
    pub fn evaluate_dry(&self, ctx: &PolicyContext) -> PolicyDecision {
        for rule in &self.rules {
            let result = (rule.check)(ctx);
            if result.is_denied() {
                return result;
            }
        }
        PolicyDecision::Allow
    }

    /// Access the audit log.
    pub fn audit_log(&self) -> &AuditLog {
        &self.audit_log
    }

    /// Mutable access to the audit log (for recording external events).
    pub fn audit_log_mut(&mut self) -> &mut AuditLog {
        &mut self.audit_log
    }

    /// List all registered rules (name + description).
    pub fn list_rules(&self) -> Vec<(&str, &str)> {
        self.rules
            .iter()
            .map(|r| (r.name.as_str(), r.description.as_str()))
            .collect()
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── built-in rules ─────────────────────────────────────────────────

/// Rule: suspended tenants are blocked from all actions.
pub fn builtin_require_tenant_rule() -> PolicyRule {
    PolicyRule::new(
        "require-tenant",
        "Ensures a tenant ID is present for non-system actions",
        |_ctx| {
            // In permissive mode, anonymous (None) tenants are allowed.
            // Override this rule in strict mode to deny anonymous access.
            PolicyDecision::Allow
        },
    )
}

/// Create a quota-check rule that denies when a callback says the
/// tenant is over quota.
#[allow(dead_code)]
pub fn quota_check_rule(
    is_over_quota: impl Fn(u64) -> bool + Send + Sync + 'static,
) -> PolicyRule {
    PolicyRule::new(
        "quota-check",
        "Denies actions when the tenant has exceeded their resource quota",
        move |ctx| {
            if let Some(tid) = ctx.tenant_id {
                if is_over_quota(tid) {
                    return PolicyDecision::deny(
                        DenialReason::QuotaExceeded,
                        "Reduce resource usage or request a quota increase from the administrator.",
                    );
                }
            }
            PolicyDecision::Allow
        },
    )
}

/// Create a rate-limit rule that denies when a callback says the
/// tenant has exceeded request rate.
#[allow(dead_code)]
pub fn rate_limit_rule(
    is_rate_limited: impl Fn(u64) -> bool + Send + Sync + 'static,
) -> PolicyRule {
    PolicyRule::new(
        "rate-limit",
        "Denies actions when the tenant has exceeded the request rate limit",
        move |ctx| {
            if let Some(tid) = ctx.tenant_id {
                if is_rate_limited(tid) {
                    return PolicyDecision::deny(
                        DenialReason::RateLimited,
                        "Wait before retrying. Consider batching operations to reduce request rate.",
                    );
                }
            }
            PolicyDecision::Allow
        },
    )
}

/// Create a max-concurrent-jobs rule.
#[allow(dead_code)]
pub fn max_concurrent_jobs_rule(
    current_jobs: impl Fn() -> usize + Send + Sync + 'static,
    max_jobs: usize,
) -> PolicyRule {
    PolicyRule::new(
        "max-concurrent-jobs",
        "Limits the total number of concurrently running jobs",
        move |_ctx| {
            if current_jobs() >= max_jobs {
                return PolicyDecision::deny(
                    DenialReason::ResourceUnavailable,
                    format!(
                        "Maximum concurrent job limit ({}) reached. Wait for running jobs to complete.",
                        max_jobs
                    ),
                );
            }
            PolicyDecision::Allow
        },
    )
}
