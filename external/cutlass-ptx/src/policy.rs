use ptx_runtime::PTXStableConfig;

/// CUTLASS GEMM TLSF allocation policy.
///
/// Tuned for GEMM workloads:
/// - 4 streams (inference is typically single-stream, extra for overlap)
/// - 90% pool fraction (GEMM workspace + activation/weight buffers)
/// - Pool health and leak detection enabled by default
#[derive(Debug, Clone, Copy)]
pub struct CutlassTlsfPolicy {
    pub max_streams: u32,
    pub pool_fraction: f32,
    pub enable_pool_health: u8,
    pub enable_leak_detection: u8,
}

impl Default for CutlassTlsfPolicy {
    fn default() -> Self {
        Self {
            max_streams: 4,
            pool_fraction: 0.90,
            enable_pool_health: 1,
            enable_leak_detection: 1,
        }
    }
}

impl CutlassTlsfPolicy {
    pub fn stable_config(self) -> PTXStableConfig {
        PTXStableConfig {
            max_streams: self.max_streams,
            pool_fraction: self.pool_fraction,
            enable_pool_health: self.enable_pool_health,
            enable_leak_detection: self.enable_leak_detection,
            ..Default::default()
        }
    }
}
