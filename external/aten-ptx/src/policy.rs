#[derive(Debug, Clone, Copy)]
pub struct InitPolicy {
    pub device_id: i32,
    pub pool_fraction: f64,
    pub num_streams: u32,
}

impl InitPolicy {
    pub const DEFAULT_NUM_STREAMS: u32 = 8;

    pub fn validate(self) -> Result<Self, String> {
        if self.device_id < 0 {
            return Err("Invalid device_id (must be >= 0)".to_string());
        }
        if !(0.1..=0.9).contains(&self.pool_fraction) {
            return Err("Invalid pool_fraction (must be 0.1-0.9)".to_string());
        }
        if self.num_streams == 0 {
            return Err("num_streams must be > 0".to_string());
        }
        Ok(self)
    }
}

pub fn ensure_libtorch_cuda_loaded() {
    let libs = ["libtorch_cpu.so", "libtorch.so", "libtorch_cuda.so"];

    for lib in libs {
        match crate::adapter::maybe_load_shared_lib(lib) {
            Ok(true) => eprintln!("[aten-ptx] loaded {}", lib),
            Ok(false) => {}
            Err(msg) => eprintln!("[aten-ptx] warning: failed to dlopen {}: {}", lib, msg),
        }
    }
}
