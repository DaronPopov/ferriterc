use std::sync::Arc;

use crate::cpu_tlsf::{global_cpu_tlsf, CpuTlsf, CpuTlsfStats};
use crate::{GpuLangRuntime, Result};

pub struct CpuCtx {
    allocator: Arc<CpuTlsf>,
}

impl CpuCtx {
    pub fn allocator(&self) -> &Arc<CpuTlsf> {
        &self.allocator
    }

    pub fn allocator_stats(&self) -> CpuTlsfStats {
        self.allocator.stats()
    }
}

pub struct GpuCtx {
    runtime: GpuLangRuntime,
}

impl GpuCtx {
    pub fn runtime(&self) -> &GpuLangRuntime {
        &self.runtime
    }

    #[cfg(feature = "torch")]
    pub fn cv(&self) -> crate::cv::CvBuilder {
        crate::cv::CvBuilder::new()
    }
}

pub fn cpu<T, E, F>(f: F) -> std::result::Result<T, E>
where
    F: FnOnce(&CpuCtx) -> std::result::Result<T, E>,
{
    let ctx = CpuCtx {
        allocator: Arc::clone(global_cpu_tlsf()),
    };
    f(&ctx)
}

pub fn gpu<T, F>(device_id: i32, f: F) -> Result<T>
where
    F: FnOnce(&GpuCtx) -> Result<T>,
{
    let ctx = GpuCtx {
        runtime: GpuLangRuntime::new(device_id)?,
    };
    f(&ctx)
}

#[cfg(feature = "torch")]
pub fn gpu_anyhow<T, F>(device_id: i32, f: F) -> anyhow::Result<T>
where
    F: FnOnce(&GpuCtx) -> anyhow::Result<T>,
{
    let ctx = GpuCtx {
        runtime: GpuLangRuntime::new(device_id).map_err(anyhow::Error::new)?,
    };
    f(&ctx)
}
