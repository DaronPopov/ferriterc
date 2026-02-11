use ptx_runtime::PTXStableConfig;

#[derive(Debug, Clone, Copy)]
pub struct XlaTlsfPolicy {
    pub max_streams: u32,
    pub pool_fraction: f32,
    pub enable_pool_health: u8,
    pub enable_leak_detection: u8,
}

impl Default for XlaTlsfPolicy {
    fn default() -> Self {
        Self {
            max_streams: 8,
            pool_fraction: 0.70,
            enable_pool_health: 1,
            enable_leak_detection: 1,
        }
    }
}

impl XlaTlsfPolicy {
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
