//! Custom compute tiling for efficient GPU operations.
//!
//! This module provides APIs for tiling compute operations across GPU blocks
//! and threads, similar to NVIDIA's CuTe library. Tiling improves cache locality
//! and enables efficient use of shared memory.
//!
//! # Tiling Concepts
//!
//! - **Tile**: A rectangular subregion of data processed together
//! - **Block Tile**: Tile size for a thread block (uses shared memory)
//! - **Thread Tile**: Tile size for a single thread (uses registers)
//! - **Grid Tile**: How tiles are distributed across the GPU grid
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use ptx_runtime::PtxRuntime;
//! use ptx_compute::tiling::{TileConfig, TiledMatmul};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let runtime = Arc::new(PtxRuntime::new(0)?);
//! let stream = runtime.stream(0);
//!
//! // Configure tiling: 128x128 block tiles, 8x8 thread tiles
//! let config = TileConfig::new()
//!     .block_tile(128, 128)
//!     .thread_tile(8, 8);
//!
//! let tiled_mm = TiledMatmul::new(&runtime, config)?;
//!
//! let (m, n, k) = (2048, 2048, 2048);
//! # let (a, b, c) = (std::ptr::null(), std::ptr::null(), std::ptr::null_mut());
//! unsafe {
//!     tiled_mm.multiply_f32(a, b, c, m, n, k, &stream)?;
//! }
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use ptx_runtime::{PtxRuntime, Stream, Result, Error};

/// Tile configuration for compute operations.
///
/// Defines how data is partitioned into tiles for processing.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Block tile dimensions (shared memory level)
    pub block_tile_m: usize,
    pub block_tile_n: usize,
    pub block_tile_k: usize,

    /// Thread tile dimensions (register level)
    pub thread_tile_m: usize,
    pub thread_tile_n: usize,

    /// Threads per block
    pub threads_x: usize,
    pub threads_y: usize,
}

impl TileConfig {
    /// Create a new tile configuration with defaults.
    ///
    /// Default: 128x128 block tiles, 8x8 thread tiles, 256 threads per block
    pub fn new() -> Self {
        Self {
            block_tile_m: 128,
            block_tile_n: 128,
            block_tile_k: 8,
            thread_tile_m: 8,
            thread_tile_n: 8,
            threads_x: 16,
            threads_y: 16,
        }
    }

    /// Set block tile dimensions.
    pub fn block_tile(mut self, m: usize, n: usize) -> Self {
        self.block_tile_m = m;
        self.block_tile_n = n;
        self
    }

    /// Set block tile K dimension (for matrix multiply).
    pub fn block_tile_k(mut self, k: usize) -> Self {
        self.block_tile_k = k;
        self
    }

    /// Set thread tile dimensions.
    pub fn thread_tile(mut self, m: usize, n: usize) -> Self {
        self.thread_tile_m = m;
        self.thread_tile_n = n;
        self
    }

    /// Set threads per block.
    pub fn threads(mut self, x: usize, y: usize) -> Self {
        self.threads_x = x;
        self.threads_y = y;
        self
    }

    /// Calculate grid dimensions for given problem size.
    pub fn grid_dims(&self, m: usize, n: usize) -> (usize, usize) {
        let blocks_m = (m + self.block_tile_m - 1) / self.block_tile_m;
        let blocks_n = (n + self.block_tile_n - 1) / self.block_tile_n;
        (blocks_m, blocks_n)
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.threads_x * self.threads_y > 1024 {
            return Err(Error::Internal {
                message: format!(
                    "Too many threads per block: {} (max 1024)",
                    self.threads_x * self.threads_y
                ),
            });
        }

        if self.block_tile_m % self.thread_tile_m != 0 {
            return Err(Error::Internal {
                message: "block_tile_m must be divisible by thread_tile_m".to_string(),
            });
        }

        if self.block_tile_n % self.thread_tile_n != 0 {
            return Err(Error::Internal {
                message: "block_tile_n must be divisible by thread_tile_n".to_string(),
            });
        }

        Ok(())
    }

    /// Get shared memory size required (in bytes).
    pub fn shared_memory_bytes(&self, elem_size: usize) -> usize {
        // Two tiles in shared memory for double buffering
        2 * (self.block_tile_m * self.block_tile_k + self.block_tile_k * self.block_tile_n) * elem_size
    }
}

impl Default for TileConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory tiling pattern.
///
/// Defines how data is accessed and stored in tiled operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilingPattern {
    /// Row-major tiling (contiguous rows)
    RowMajor,
    /// Column-major tiling (contiguous columns)
    ColumnMajor,
    /// Swizzled tiling (reduces bank conflicts)
    Swizzled,
}

/// Tiled matrix multiplication.
///
/// Performs matrix multiplication using custom tiling for optimal
/// performance. Uses shared memory and register tiling.
pub struct TiledMatmul {
    #[allow(dead_code)]
    runtime: Arc<PtxRuntime>,
    config: TileConfig,
}

impl TiledMatmul {
    /// Create a new tiled matmul operation.
    pub fn new(runtime: &Arc<PtxRuntime>, config: TileConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            runtime: Arc::clone(runtime),
            config,
        })
    }

    /// Get the tile configuration.
    pub fn config(&self) -> &TileConfig {
        &self.config
    }

    /// Perform tiled matrix multiplication: C = A @ B
    ///
    /// DEFERRED: Custom tiled kernel not yet available in ptx-sys.
    /// Use `ptx_compute::gemm::Matmul` (cuBLAS) for production matmul.
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A (m × k)
    /// * `b` - Matrix B (k × n)
    /// * `c` - Output matrix C (m × n)
    /// * `m, n, k` - Matrix dimensions
    /// * `stream` - CUDA stream
    ///
    /// # Safety
    ///
    /// Pointers must be valid GPU memory.
    pub unsafe fn multiply_f32(
        &self,
        _a: *const f32,
        _b: *const f32,
        _c: *mut f32,
        m: usize,
        n: usize,
        _k: usize,
        _stream: &Stream,
    ) -> Result<()> {
        // Calculate grid dimensions
        let (grid_m, grid_n) = self.config.grid_dims(m, n);

        // DEFERRED: No ptx_sys tiled matmul kernel exists yet.
        // Use ptx_compute::gemm::Matmul (cuBLAS-backed) instead.
        let _ = (grid_m, grid_n);
        Err(Error::NotSupported {
            message: "Tiled matmul kernel not yet available in ptx-sys. Use gemm::Matmul instead.".to_string(),
        })
    }

    /// Get estimated FLOPS for this tiled operation.
    pub fn flops(m: usize, n: usize, k: usize) -> f64 {
        2.0 * m as f64 * n as f64 * k as f64
    }

    /// Get expected shared memory usage.
    pub fn shared_memory_bytes(&self) -> usize {
        self.config.shared_memory_bytes(std::mem::size_of::<f32>())
    }

    /// Get grid dimensions for problem size.
    pub fn grid_dims(&self, m: usize, n: usize) -> (usize, usize) {
        self.config.grid_dims(m, n)
    }
}

/// Tile iterator for processing large tensors in chunks.
///
/// Useful for operations that don't fit in GPU memory - process
/// one tile at a time.
pub struct TileIterator {
    total_m: usize,
    total_n: usize,
    tile_m: usize,
    tile_n: usize,
    current_m: usize,
    current_n: usize,
}

impl TileIterator {
    /// Create a new tile iterator.
    ///
    /// # Arguments
    ///
    /// * `total_m, total_n` - Total problem dimensions
    /// * `tile_m, tile_n` - Tile dimensions
    pub fn new(total_m: usize, total_n: usize, tile_m: usize, tile_n: usize) -> Self {
        Self {
            total_m,
            total_n,
            tile_m,
            tile_n,
            current_m: 0,
            current_n: 0,
        }
    }

    /// Get the next tile.
    ///
    /// Returns (start_m, start_n, extent_m, extent_n) or None if done.
    pub fn next(&mut self) -> Option<(usize, usize, usize, usize)> {
        if self.current_m >= self.total_m {
            return None;
        }

        let start_m = self.current_m;
        let start_n = self.current_n;
        let extent_m = (self.tile_m).min(self.total_m - start_m);
        let extent_n = (self.tile_n).min(self.total_n - start_n);

        // Advance to next tile
        self.current_n += self.tile_n;
        if self.current_n >= self.total_n {
            self.current_n = 0;
            self.current_m += self.tile_m;
        }

        Some((start_m, start_n, extent_m, extent_n))
    }

    /// Reset iterator to beginning.
    pub fn reset(&mut self) {
        self.current_m = 0;
        self.current_n = 0;
    }

    /// Get total number of tiles.
    pub fn num_tiles(&self) -> usize {
        let tiles_m = (self.total_m + self.tile_m - 1) / self.tile_m;
        let tiles_n = (self.total_n + self.tile_n - 1) / self.tile_n;
        tiles_m * tiles_n
    }
}

/// 2D tiling helper for arbitrary operations.
///
/// Generic tiling interface for custom operations.
pub struct Tile2D {
    pub row_start: usize,
    pub col_start: usize,
    pub row_extent: usize,
    pub col_extent: usize,
}

impl Tile2D {
    /// Create a new 2D tile.
    pub fn new(row_start: usize, col_start: usize, row_extent: usize, col_extent: usize) -> Self {
        Self {
            row_start,
            col_start,
            row_extent,
            col_extent,
        }
    }

    /// Get the number of elements in this tile.
    pub fn size(&self) -> usize {
        self.row_extent * self.col_extent
    }

    /// Check if a point is within this tile.
    pub fn contains(&self, row: usize, col: usize) -> bool {
        row >= self.row_start
            && row < self.row_start + self.row_extent
            && col >= self.col_start
            && col < self.col_start + self.col_extent
    }

    /// Get offset for a point within the tile.
    pub fn offset(&self, row: usize, col: usize, stride: usize) -> Option<usize> {
        if self.contains(row, col) {
            let local_row = row - self.row_start;
            let local_col = col - self.col_start;
            Some(local_row * stride + local_col)
        } else {
            None
        }
    }
}

/// Generate optimal tile configuration for problem size.
///
/// Analyzes problem dimensions and GPU capabilities to suggest
/// optimal tiling parameters.
pub fn suggest_tile_config(m: usize, n: usize, k: usize) -> TileConfig {
    // Simple heuristics - could be made more sophisticated
    let config = TileConfig::new();

    // For small problems, use smaller tiles
    if m < 512 && n < 512 {
        return config.block_tile(64, 64).thread_tile(8, 8);
    }

    // For large problems with large K, increase K tiling
    if k > 2048 {
        return config.block_tile(128, 128).block_tile_k(16);
    }

    // Default is already good for most cases
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_validation() {
        let config = TileConfig::new();
        assert!(config.validate().is_ok());

        let bad_config = TileConfig::new().threads(64, 64); // 4096 threads > 1024
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_tile_iterator() {
        let mut iter = TileIterator::new(100, 100, 32, 32);

        let mut count = 0;
        while let Some((_, _, _, _)) = iter.next() {
            count += 1;
        }

        assert_eq!(count, 16); // 4x4 grid of tiles
        assert_eq!(iter.num_tiles(), 16);
    }

    #[test]
    fn test_tile_2d() {
        let tile = Tile2D::new(10, 20, 5, 5);

        assert!(tile.contains(10, 20));
        assert!(tile.contains(14, 24));
        assert!(!tile.contains(15, 20));
        assert!(!tile.contains(10, 25));

        assert_eq!(tile.size(), 25);
    }

    #[test]
    fn test_grid_dims() {
        let config = TileConfig::new().block_tile(128, 128);

        let (grid_m, grid_n) = config.grid_dims(1000, 1000);
        assert_eq!(grid_m, 8); // ceil(1000/128)
        assert_eq!(grid_n, 8);

        let (grid_m2, grid_n2) = config.grid_dims(128, 256);
        assert_eq!(grid_m2, 1);
        assert_eq!(grid_n2, 2);
    }
}
