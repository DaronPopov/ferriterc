//! Policy decision model for the Ferrite OS control plane.
//!
//! Every policy evaluation produces a [`PolicyDecision`]: either [`Allow`] or
//! [`Deny`] with a machine-readable reason code and human-friendly remediation
//! text.  Consumers never need to guess why an action was rejected.

use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Outcome of a policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// The action is permitted.
    Allow,
    /// The action is denied with a structured reason and remediation advice.
    Deny {
        reason: DenialReason,
        remediation: String,
    },
}

#[allow(dead_code)]
impl PolicyDecision {
    /// Returns `true` when the decision permits the action.
    pub fn is_allowed(&self) -> bool {
        matches!(self, PolicyDecision::Allow)
    }

    /// Returns `true` when the decision blocks the action.
    pub fn is_denied(&self) -> bool {
        matches!(self, PolicyDecision::Deny { .. })
    }

    /// Convenience constructor for a denial.
    pub fn deny(reason: DenialReason, remediation: impl Into<String>) -> Self {
        PolicyDecision::Deny {
            reason,
            remediation: remediation.into(),
        }
    }
}

impl fmt::Display for PolicyDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyDecision::Allow => write!(f, "ALLOW"),
            PolicyDecision::Deny { reason, remediation } => {
                write!(f, "DENY [{}]: {}", reason, remediation)
            }
        }
    }
}

/// Machine-readable reason for a policy denial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DenialReason {
    /// Tenant has exceeded their resource quota (VRAM, streams, etc.).
    QuotaExceeded,
    /// Request rate limit has been hit.
    RateLimited,
    /// The tenant account is suspended or disabled.
    TenantSuspended,
    /// The requested resource is temporarily unavailable.
    ResourceUnavailable,
    /// The caller lacks authorization for this action.
    Unauthorized,
    /// A custom policy rule was violated.
    PolicyViolation(String),
}

impl DenialReason {
    /// Return a stable diagnostic code for this denial reason.
    pub fn code(&self) -> &'static str {
        match self {
            DenialReason::QuotaExceeded => "POLICY-DENY-0001",
            DenialReason::RateLimited => "POLICY-DENY-0002",
            DenialReason::TenantSuspended => "POLICY-DENY-0003",
            DenialReason::ResourceUnavailable => "POLICY-DENY-0004",
            DenialReason::Unauthorized => "POLICY-DENY-0005",
            DenialReason::PolicyViolation(_) => "POLICY-DENY-0006",
        }
    }
}

impl fmt::Display for DenialReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenialReason::QuotaExceeded => write!(f, "QuotaExceeded"),
            DenialReason::RateLimited => write!(f, "RateLimited"),
            DenialReason::TenantSuspended => write!(f, "TenantSuspended"),
            DenialReason::ResourceUnavailable => write!(f, "ResourceUnavailable"),
            DenialReason::Unauthorized => write!(f, "Unauthorized"),
            DenialReason::PolicyViolation(detail) => write!(f, "PolicyViolation({})", detail),
        }
    }
}

/// Context supplied to every policy evaluation.
///
/// Describes the tenant, the action being attempted, the target resource,
/// and the wall-clock time of the request.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PolicyContext {
    /// Tenant performing the action (None for unauthenticated/system).
    pub tenant_id: Option<u64>,
    /// Human-readable action name (e.g. "app-start", "kill-job").
    pub action: String,
    /// Target resource identifier (e.g. app name, job id).
    pub resource: String,
    /// When the request was received.
    pub timestamp: Instant,
}

impl PolicyContext {
    /// Create a new context with the current timestamp.
    pub fn new(tenant_id: Option<u64>, action: impl Into<String>, resource: impl Into<String>) -> Self {
        Self {
            tenant_id,
            action: action.into(),
            resource: resource.into(),
            timestamp: Instant::now(),
        }
    }
}
