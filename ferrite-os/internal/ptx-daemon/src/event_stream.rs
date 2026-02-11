//! Machine-readable event stream for the Ferrite OS control plane.
//!
//! Provides a sequenced, JSON-serializable stream of scheduler and policy
//! events.  Consumers can subscribe for real-time delivery or query
//! historical events by sequence number.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Event types emitted by the scheduler event stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SchedulerEvent {
    /// A job was added to the queue.
    JobQueued {
        job_id: u64,
        tenant_id: u64,
        priority: i32,
    },
    /// A job started executing on the GPU.
    JobStarted {
        job_id: u64,
        tenant_id: u64,
    },
    /// A job completed successfully.
    JobCompleted {
        job_id: u64,
        tenant_id: u64,
        elapsed_ms: u64,
    },
    /// A job failed.
    JobFailed {
        job_id: u64,
        tenant_id: u64,
        error: String,
    },
    /// A resource quota was changed.
    QuotaChanged {
        tenant_id: u64,
        resource: String,
        old_limit: u64,
        new_limit: u64,
    },
    /// A policy decision was made.
    PolicyDecision {
        tenant_id: u64,
        action: String,
        decision: String,
        reason: Option<String>,
    },
    /// The scheduler queue was paused.
    QueuePaused,
    /// The scheduler queue was resumed.
    QueueResumed,
    /// An audit query was performed.
    AuditQuery {
        tenant_filter: Option<u64>,
        result_count: usize,
    },
}

/// A single entry in the event stream, with a monotonic sequence number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStreamEntry {
    /// Monotonically increasing sequence number.
    pub sequence_number: u64,
    /// Elapsed seconds since stream creation.
    pub timestamp_secs: f64,
    /// The event type and payload.
    #[serde(flatten)]
    pub event: SchedulerEvent,
}

/// The event stream: records events in a bounded buffer and supports
/// fan-out to subscribers via channels.
pub struct SchedulerEventStream {
    entries: VecDeque<EventStreamEntry>,
    max_entries: usize,
    next_seq: u64,
    boot_time: Instant,
    subscribers: Vec<Sender<EventStreamEntry>>,
}

#[allow(dead_code)]
impl SchedulerEventStream {
    /// Create a new event stream with the given buffer capacity.
    pub fn new(buffer_size: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(buffer_size.min(100_000)),
            max_entries: buffer_size,
            next_seq: 1,
            boot_time: Instant::now(),
            subscribers: Vec::new(),
        }
    }

    /// Emit a new event into the stream.
    pub fn emit(&mut self, event: SchedulerEvent) {
        let entry = EventStreamEntry {
            sequence_number: self.next_seq,
            timestamp_secs: self.boot_time.elapsed().as_secs_f64(),
            event,
        };
        self.next_seq += 1;

        // Fan out to live subscribers, removing disconnected ones.
        self.subscribers
            .retain(|tx| tx.send(entry.clone()).is_ok());

        // Buffer the entry.
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Subscribe to the event stream.  Returns a receiver that will
    /// get all future events until the sender is dropped.
    pub fn subscribe(&mut self) -> Receiver<EventStreamEntry> {
        let (tx, rx) = mpsc::channel();
        self.subscribers.push(tx);
        rx
    }

    /// Export all entries since (and including) the given sequence number.
    pub fn export_since(&self, since_seq: u64) -> Vec<&EventStreamEntry> {
        self.entries
            .iter()
            .filter(|e| e.sequence_number >= since_seq)
            .collect()
    }

    /// Export the last N entries.
    pub fn recent(&self, n: usize) -> Vec<&EventStreamEntry> {
        self.entries.iter().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect()
    }

    /// Total entries currently buffered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The next sequence number that will be assigned.
    pub fn next_sequence(&self) -> u64 {
        self.next_seq
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Export all buffered entries as JSON lines (one JSON object per line).
    pub fn export_jsonl(&self) -> String {
        self.entries
            .iter()
            .filter_map(|e| serde_json::to_string(e).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
