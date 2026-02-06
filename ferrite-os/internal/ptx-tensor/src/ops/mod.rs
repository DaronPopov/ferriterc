//! Tensor operations.
//!
//! This module contains all tensor operations organized by category:
//! - Binary operations (add, mul, sub, div, etc.)
//! - Unary operations (exp, log, sqrt, etc.)
//! - Activation functions (relu, gelu, sigmoid, etc.)
//! - Reduction operations (sum, mean, max, etc.)
//! - Softmax operations
//! - Matrix operations (matmul via cuBLAS)

pub mod binary;
pub mod unary;
pub mod activation;
pub mod reduction;
pub mod softmax;
pub mod matmul;

// Re-export all operations
pub use binary::*;
pub use unary::*;
pub use activation::*;
pub use reduction::*;
pub use matmul::*;
