//! Multi-tenant GPU scheduler for Ferrite-OS.
//!
//! This module provides a complete scheduling subsystem for managing GPU workloads
//! across multiple tenants with resource isolation, fair dispatch, and quota enforcement.
//!
//! # Architecture
//!
//! The scheduler is composed of several collaborating components:
//!
//! - **Tenants** ([`tenant`]): Identity, quotas, and live usage tracking per workload owner.
//! - **Jobs** ([`job`]): Discrete GPU work units with a well-defined lifecycle state machine.
//! - **Policies** ([`policy`]): Pluggable ordering and admission strategies (FIFO, fair-share).
//! - **Registry** ([`registry`]): Central tenant registry with quota management.
//! - **Dispatcher** ([`dispatcher`]): Stream assignment, backpressure, and active job tracking.
//! - **Diagnostics** ([`diagnostics`]): Structured event logging and health monitoring.
//!
//! # Backward Compatibility
//!
//! The default tenant (id=0, unlimited quotas) ensures that existing single-user
//! workloads continue to function without any scheduler configuration. New tenant-aware
//! APIs are purely additive.
//!
//! # Example
//!
//! ```ignore
//! use ptx_runtime::scheduler::*;
//!
//! // Create a scheduler with fair-share policy
//! let mut scheduler = Scheduler::new(SchedulerConfig::default());
//!
//! // Register tenants
//! scheduler.register_tenant(TenantId(1), "inference", TenantQuotas {
//!     max_vram_bytes: 4 * 1024 * 1024 * 1024,
//!     max_streams: 8,
//!     max_concurrent_jobs: 16,
//!     max_runtime_budget_ms: u64::MAX,
//! });
//!
//! // Submit and dispatch jobs
//! let job = Job::new(TenantId(1), StreamPriority::Normal);
//! scheduler.submit(job).unwrap();
//! let dispatched = scheduler.run_cycle(&pool);
//! ```

pub mod tenant;
pub mod job;
pub mod policy;
pub mod registry;
pub mod dispatcher;
pub mod diagnostics;

// Re-export primary types for convenience.
pub use tenant::{Tenant, TenantId, TenantQuotas, TenantUsage, TenantUsageSnapshot};
pub use job::{Job, JobId, JobState, InvalidTransition};
pub use policy::{
    AdmissionDecision, DenialReason, FairSharePolicy, FifoPolicy, SchedulerPolicy,
};
pub use registry::{TenantRegistry, RegistryError};
pub use dispatcher::{DispatchRecord, Dispatcher};
pub use diagnostics::{SchedulerDiagnostics, SchedulerEvent};

use crate::error::Result;
use crate::stream::StreamPool;

/// High-level scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Scheduling policy name: "fair-share" or "fifo".
    pub policy: String,
    /// Default quotas applied to newly registered tenants.
    pub default_tenant_quotas: TenantQuotas,
    /// Maximum number of jobs that can be queued in the dispatcher.
    pub max_queued_jobs: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: "fair-share".to_string(),
            default_tenant_quotas: TenantQuotas::unlimited(),
            max_queued_jobs: 4096,
        }
    }
}

/// The top-level scheduler that composes the registry, policy, and dispatcher.
///
/// This is the primary entry point for multi-tenant scheduling. It owns the
/// tenant registry, the chosen policy, and the job dispatcher.
pub struct Scheduler {
    registry: TenantRegistry,
    policy: Box<dyn SchedulerPolicy>,
    dispatcher: Dispatcher,
    config: SchedulerConfig,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        let policy: Box<dyn SchedulerPolicy> = match config.policy.as_str() {
            "fifo" => Box::new(FifoPolicy),
            _ => Box::new(FairSharePolicy::new()),
        };

        let dispatcher = Dispatcher::new(config.max_queued_jobs);

        tracing::info!(
            policy = policy.name(),
            max_queued_jobs = config.max_queued_jobs,
            "scheduler initialized"
        );

        Self {
            registry: TenantRegistry::new(),
            policy,
            dispatcher,
            config,
        }
    }

    /// Register a new tenant with specific quotas.
    pub fn register_tenant(
        &mut self,
        id: TenantId,
        label: impl Into<String>,
        quotas: TenantQuotas,
    ) -> std::result::Result<(), RegistryError> {
        self.registry.register(id, label, quotas)?;
        Ok(())
    }

    /// Register a new tenant with default quotas from the scheduler config.
    pub fn register_tenant_default(
        &mut self,
        id: TenantId,
        label: impl Into<String>,
    ) -> std::result::Result<(), RegistryError> {
        self.registry
            .register(id, label, self.config.default_tenant_quotas.clone())?;
        Ok(())
    }

    /// Submit a job to the scheduler queue.
    pub fn submit(&mut self, job: Job) -> Result<JobId> {
        self.dispatcher.submit(job)
    }

    /// Run one scheduling cycle: order, admit, dispatch queued jobs.
    pub fn run_cycle(&mut self, pool: &StreamPool) -> Vec<DispatchRecord> {
        self.dispatcher
            .dispatch_cycle(pool, self.policy.as_ref(), &self.registry)
    }

    /// Mark a job as completed.
    pub fn complete_job(&mut self, job_id: JobId) -> Option<DispatchRecord> {
        self.dispatcher.complete_job(job_id, &self.registry)
    }

    /// Mark a job as failed.
    pub fn fail_job(&mut self, job_id: JobId, reason: &str) -> Option<DispatchRecord> {
        self.dispatcher.fail_job(job_id, reason, &self.registry)
    }

    /// Get a reference to the tenant registry.
    pub fn registry(&self) -> &TenantRegistry {
        &self.registry
    }

    /// Get a mutable reference to the tenant registry.
    pub fn registry_mut(&mut self) -> &mut TenantRegistry {
        &mut self.registry
    }

    /// Get a reference to the dispatcher.
    pub fn dispatcher(&self) -> &Dispatcher {
        &self.dispatcher
    }

    /// Get the active scheduling policy name.
    pub fn policy_name(&self) -> &'static str {
        self.policy.name()
    }

    /// Get the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Update tenant VRAM usage (used by memory allocation hooks).
    pub fn update_vram_usage(&self, tenant_id: TenantId, delta: i64) -> Option<u64> {
        self.registry.update_vram_usage(tenant_id, delta)
    }

    /// Check if the stream pool is saturated.
    pub fn is_saturated(&self, pool: &StreamPool) -> bool {
        self.dispatcher.is_saturated(pool)
    }

    /// Get the scheduler health diagnostic.
    pub fn health_diagnostic(&self) -> crate::telemetry::DiagnosticEvent {
        self.dispatcher.diagnostics().health_diagnostic()
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("policy", &self.policy.name())
            .field("registry", &self.registry)
            .field("dispatcher", &self.dispatcher)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{Stream, StreamPriority};

    fn mock_pool(n: usize) -> StreamPool {
        let streams: Vec<Stream> = (0..n)
            .map(|i| Stream::new(std::ptr::null_mut(), i as i32))
            .collect();
        StreamPool::new(streams)
    }

    #[test]
    fn test_scheduler_default_config() {
        let config = SchedulerConfig::default();
        assert_eq!(config.policy, "fair-share");
        assert_eq!(config.max_queued_jobs, 4096);
    }

    #[test]
    fn test_scheduler_lifecycle() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let pool = mock_pool(4);

        // Default tenant exists
        assert!(sched.registry().get(TenantId::DEFAULT).is_some());

        // Submit a job for the default tenant
        let job = Job::new(TenantId::DEFAULT, StreamPriority::Normal);
        let job_id = sched.submit(job).unwrap();

        // Run dispatch cycle
        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 1);
        assert_eq!(dispatched[0].job_id, job_id);

        // Complete the job
        let record = sched.complete_job(job_id).unwrap();
        assert_eq!(record.job_id, job_id);
    }

    #[test]
    fn test_scheduler_multi_tenant() {
        let config = SchedulerConfig {
            policy: "fair-share".to_string(),
            default_tenant_quotas: TenantQuotas::unlimited(),
            max_queued_jobs: 100,
        };
        let mut sched = Scheduler::new(config);
        let pool = mock_pool(8);

        sched
            .register_tenant(
                TenantId(1),
                "training",
                TenantQuotas {
                    max_vram_bytes: u64::MAX,
                    max_streams: 4,
                    max_concurrent_jobs: 10,
                    max_runtime_budget_ms: u64::MAX,
                },
            )
            .unwrap();
        sched
            .register_tenant(
                TenantId(2),
                "inference",
                TenantQuotas {
                    max_vram_bytes: u64::MAX,
                    max_streams: 4,
                    max_concurrent_jobs: 10,
                    max_runtime_budget_ms: u64::MAX,
                },
            )
            .unwrap();

        // Submit jobs from both tenants
        for _ in 0..3 {
            sched
                .submit(Job::new(TenantId(1), StreamPriority::Normal))
                .unwrap();
            sched
                .submit(Job::new(TenantId(2), StreamPriority::Normal))
                .unwrap();
        }

        let dispatched = sched.run_cycle(&pool);
        assert_eq!(dispatched.len(), 6);

        let t1_count = dispatched.iter().filter(|r| r.tenant_id == TenantId(1)).count();
        let t2_count = dispatched.iter().filter(|r| r.tenant_id == TenantId(2)).count();
        assert_eq!(t1_count, 3);
        assert_eq!(t2_count, 3);
    }

    #[test]
    fn test_scheduler_fifo_policy() {
        let config = SchedulerConfig {
            policy: "fifo".to_string(),
            default_tenant_quotas: TenantQuotas::unlimited(),
            max_queued_jobs: 100,
        };
        let sched = Scheduler::new(config);
        assert_eq!(sched.policy_name(), "fifo");
    }

    #[test]
    fn test_scheduler_vram_tracking() {
        let sched = Scheduler::new(SchedulerConfig::default());

        let new_usage = sched.update_vram_usage(TenantId::DEFAULT, 4096);
        assert_eq!(new_usage, Some(4096));

        let new_usage = sched.update_vram_usage(TenantId::DEFAULT, -1024);
        assert_eq!(new_usage, Some(3072));
    }

    #[test]
    fn test_scheduler_health() {
        let sched = Scheduler::new(SchedulerConfig::default());
        let health = sched.health_diagnostic();
        assert_eq!(health.status, crate::telemetry::DiagnosticStatus::PASS);
    }

    #[test]
    fn test_register_tenant_default_quotas() {
        let config = SchedulerConfig {
            policy: "fifo".to_string(),
            default_tenant_quotas: TenantQuotas {
                max_vram_bytes: 1_000_000,
                max_streams: 2,
                max_concurrent_jobs: 5,
                max_runtime_budget_ms: 30_000,
            },
            max_queued_jobs: 100,
        };
        let mut sched = Scheduler::new(config);
        sched.register_tenant_default(TenantId(10), "auto-tenant").unwrap();

        let tenant = sched.registry().get(TenantId(10)).unwrap();
        assert_eq!(tenant.quotas.max_vram_bytes, 1_000_000);
        assert_eq!(tenant.quotas.max_streams, 2);
    }
}
