//! Runtime statistics and operation counting.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global operation counter for tracking tensor operations.
static OPS_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Increment the operation counter.
#[inline]
pub fn increment_ops() {
    OPS_COUNTER.fetch_add(1, Ordering::Relaxed);
}

/// Increment the operation counter by a specific amount.
#[inline]
pub fn increment_ops_by(n: u64) {
    OPS_COUNTER.fetch_add(n, Ordering::Relaxed);
}

/// Get the current operation count.
#[inline]
pub fn get_ops_count() -> u64 {
    OPS_COUNTER.load(Ordering::Relaxed)
}

/// Reset the operation counter to zero.
#[inline]
pub fn reset_ops_count() {
    OPS_COUNTER.store(0, Ordering::Relaxed);
}
