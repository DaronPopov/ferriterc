// CUTLASS GEMM kernel instantiations with C-linkage launchers.
//
// Provides three kernel families:
//   1. FP16 x FP16 -> FP16 GEMM (tensor core, FP32 accumulator)
//   2. INT8 x INT8 -> FP16 GEMM (tensor core, INT32 accumulator, FP16 output via epilogue)
//   3. INT4 x INT4 -> FP16 GEMM (tensor core, INT32 accumulator, FP16 output via epilogue)
//
// Kernels 2 and 3 are designed for quantized inference: both activations and
// weights are quantized, the INT32 accumulator preserves precision, and the
// epilogue applies a dequantization scale (alpha) and converts to FP16 output.
//
// All kernels target sm_80+ (Ampere tensor cores, covers Orin sm_87).

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cstdio>
#include <cstdint>

// ============================================================================
// Kernel 1: FP16 x FP16 -> FP16 GEMM (tensor core)
//
// Layout: A = RowMajor (M,K), B = ColumnMajor (K,N transposed), C/D = RowMajor (M,N)
// Accumulator: FP32 for numerical stability
// MMA instruction: m16n8k16 (Ampere FP16 tensor core)
// ============================================================================

using EpilogueOp_F16 = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,                     // output element type
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,  // elements per access (8)
    float,                               // accumulator type
    float                                // compute type for epilogue
>;

using GemmF16 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                      // ElementA
    cutlass::layout::RowMajor,            // LayoutA
    cutlass::half_t,                      // ElementB
    cutlass::layout::ColumnMajor,         // LayoutB (transposed weight)
    cutlass::half_t,                      // ElementC
    cutlass::layout::RowMajor,            // LayoutC
    float,                                // Accumulator
    cutlass::arch::OpClassTensorOp,       // Tensor cores
    cutlass::arch::Sm80,                  // Ampere (covers sm_87)
    cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,     // MMA instruction (Ampere FP16)
    EpilogueOp_F16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                                     // Pipeline stages
>;

extern "C" {

/// Query workspace size for FP16 GEMM.
size_t cutlass_hgemm_workspace_size(int M, int N, int K) {
    GemmF16::Arguments args(
        {M, N, K},
        {nullptr, K},    // A (M,K) row-major -> lda=K
        {nullptr, K},    // B (N,K) col-major stored as (K,N) -> ldb=K
        {nullptr, N},    // C (M,N) -> ldc=N
        {nullptr, N},    // D (M,N) -> ldd=N
        {1.0f, 0.0f}
    );
    return GemmF16::get_workspace_size(args);
}

/// Launch FP16 x FP16 -> FP16 GEMM.
///
/// A: (M, K) row-major FP16
/// B: (N, K) stored column-major (i.e. (K, N) transposed) FP16
/// C: (M, N) row-major FP16 output
///
/// Returns 0 on success, non-zero cutlass::Status on failure.
int cutlass_hgemm(
    const void *A, const void *B, void *C,
    int M, int N, int K,
    float alpha, float beta,
    void *workspace, size_t workspace_size
) {
    GemmF16 gemm_op;

    GemmF16::Arguments args(
        {M, N, K},
        {static_cast<const cutlass::half_t*>(A), K},     // A: lda=K
        {static_cast<const cutlass::half_t*>(B), K},     // B: ldb=K
        {static_cast<const cutlass::half_t*>(C), N},     // C: ldc=N (in-place ok)
        {static_cast<cutlass::half_t*>(C), N},           // D: ldd=N
        {alpha, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] can_implement failed: %d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] initialize failed: %d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] run failed: %d\n", (int)status);
    }

    return (int)status;
}

// ============================================================================
// Kernel 1b: FP16 x FP16 -> FP16 GEMM (column-major, cuBLAS-compatible)
//
// Layout: A = ColumnMajor (M,K), B = ColumnMajor (K,N), C/D = ColumnMajor (M,N)
// This matches cuBLAS convention directly — no transpose/swap needed.
// Accumulator: FP32 for numerical stability
// MMA instruction: m16n8k16 (Ampere FP16 tensor core)
// ============================================================================

using GemmF16_NN = cutlass::gemm::device::Gemm<
    cutlass::half_t,                      // ElementA
    cutlass::layout::ColumnMajor,         // LayoutA (col-major, cuBLAS convention)
    cutlass::half_t,                      // ElementB
    cutlass::layout::ColumnMajor,         // LayoutB (col-major, cuBLAS convention)
    cutlass::half_t,                      // ElementC
    cutlass::layout::ColumnMajor,         // LayoutC (col-major, cuBLAS convention)
    float,                                // Accumulator
    cutlass::arch::OpClassTensorOp,       // Tensor cores
    cutlass::arch::Sm80,                  // Ampere (covers sm_87)
    cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,     // MMA instruction (Ampere FP16)
    EpilogueOp_F16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                                     // Pipeline stages
>;

/// Query workspace size for column-major FP16 GEMM.
size_t cutlass_hgemm_nn_workspace_size(int M, int N, int K) {
    GemmF16_NN::Arguments args(
        {M, N, K},
        {nullptr, M},    // A (M,K) col-major -> lda=M
        {nullptr, K},    // B (K,N) col-major -> ldb=K
        {nullptr, M},    // C (M,N) col-major -> ldc=M
        {nullptr, M},    // D (M,N) col-major -> ldd=M
        {1.0f, 0.0f}
    );
    return GemmF16_NN::get_workspace_size(args);
}

/// Launch cuBLAS-compatible column-major FP16 x FP16 -> FP16 GEMM.
///
/// A: (M, K) column-major FP16, leading dimension lda
/// B: (K, N) column-major FP16, leading dimension ldb
/// C: (M, N) column-major FP16 output, leading dimension ldc
///
/// Accepts lda/ldb/ldc directly from cuBLAS (no transpose gymnastics).
/// Returns 0 on success, non-zero cutlass::Status on failure.
int cutlass_hgemm_nn(
    const void *A, const void *B, void *C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta,
    void *workspace, size_t workspace_size
) {
    GemmF16_NN gemm_op;

    GemmF16_NN::Arguments args(
        {M, N, K},
        {static_cast<const cutlass::half_t*>(A), lda},   // A: col-major, lda
        {static_cast<const cutlass::half_t*>(B), ldb},   // B: col-major, ldb
        {static_cast<const cutlass::half_t*>(C), ldc},   // C: col-major, ldc (in-place ok)
        {static_cast<cutlass::half_t*>(C), ldc},          // D: col-major, ldc
        {alpha, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] nn can_implement failed: %d (M=%d N=%d K=%d)\n",
                (int)status, M, N, K);
        return (int)status;
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] nn initialize failed: %d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] nn run failed: %d\n", (int)status);
    }

    return (int)status;
}

// ============================================================================
// Kernel 2: INT8 x INT8 -> FP16 quantized GEMM
//
// For Q8_0 quantized inference: both activations and weights are INT8.
// INT32 accumulator preserves precision; epilogue applies dequant scale
// (via alpha) and converts to FP16 output.
//
// Layout: A = RowMajor INT8 (M,K), B = ColumnMajor INT8 (K,N transposed)
// Output: FP16 (M,N) row-major
// MMA instruction: m16n8k32 (Ampere INT8 tensor core)
// ============================================================================

using EpilogueOp_I32toF16 = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,                      // output element type (FP16)
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,  // elements per access (8)
    int32_t,                              // accumulator type
    float                                 // compute type for epilogue scale/bias
>;

using GemmI8 = cutlass::gemm::device::Gemm<
    int8_t,                               // ElementA (quantized activations)
    cutlass::layout::RowMajor,            // LayoutA
    int8_t,                               // ElementB (quantized weights)
    cutlass::layout::ColumnMajor,         // LayoutB
    cutlass::half_t,                      // ElementC (FP16 output)
    cutlass::layout::RowMajor,            // LayoutC
    int32_t,                              // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,  // Threadblock tile
    cutlass::gemm::GemmShape<64, 64, 64>,    // Warp tile
    cutlass::gemm::GemmShape<16, 8, 32>,     // MMA instruction (Ampere INT8)
    EpilogueOp_I32toF16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,                                    // Pipeline stages
    16,                                   // AlignmentA: 128 bits / 8 bits = 16
    16                                    // AlignmentB: 128 bits / 8 bits = 16
>;

/// Query workspace size for INT8 GEMM.
size_t cutlass_gemm_i8_workspace_size(int M, int N, int K) {
    GemmI8::Arguments args(
        {M, N, K},
        {nullptr, K},
        {nullptr, K},
        {nullptr, N},
        {nullptr, N},
        {1.0f, 0.0f}
    );
    return GemmI8::get_workspace_size(args);
}

/// Launch INT8 x INT8 -> FP16 GEMM.
///
/// A: (M, K) row-major INT8 quantized activations
/// B: (N, K) column-major INT8 quantized weights
/// C: (M, N) row-major FP16 output
/// scale: dequantization scale applied in epilogue (alpha = scale_a * scale_b)
///
/// Returns 0 on success.
int cutlass_gemm_i8(
    const void *A, const void *B, void *C,
    int M, int N, int K,
    float scale, float beta,
    void *workspace, size_t workspace_size
) {
    GemmI8 gemm_op;

    GemmI8::Arguments args(
        {M, N, K},
        {static_cast<const int8_t*>(A), K},
        {static_cast<const int8_t*>(B), K},
        {static_cast<const cutlass::half_t*>(C), N},
        {static_cast<cutlass::half_t*>(C), N},
        {scale, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] i8 can_implement failed: %d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] i8 initialize failed: %d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] i8 run failed: %d\n", (int)status);
    }

    return (int)status;
}

// ============================================================================
// Kernel 3: INT4 x INT4 -> FP16 quantized GEMM
//
// For Q4_0 quantized inference: both activations and weights are INT4.
// Uses CUTLASS int4b_t sub-byte type (packed, 2 elements per byte).
// INT32 accumulator; epilogue applies dequant scale and converts to FP16.
//
// Layout: A = RowMajor INT4 (M,K), B = ColumnMajor INT4 (K,N transposed)
// Output: FP16 (M,N) row-major
// MMA instruction: m16n8k64 (Ampere INT4 tensor core)
// ============================================================================

using GemmI4 = cutlass::gemm::device::Gemm<
    cutlass::int4b_t,                     // ElementA (4-bit quantized activations)
    cutlass::layout::RowMajor,            // LayoutA
    cutlass::int4b_t,                     // ElementB (4-bit quantized weights)
    cutlass::layout::ColumnMajor,         // LayoutB
    cutlass::half_t,                      // ElementC (FP16 output)
    cutlass::layout::RowMajor,            // LayoutC
    int32_t,                              // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>, // Threadblock tile
    cutlass::gemm::GemmShape<64, 64, 128>,   // Warp tile
    cutlass::gemm::GemmShape<16, 8, 64>,     // MMA instruction (Ampere INT4)
    EpilogueOp_I32toF16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,                                    // Pipeline stages
    32,                                   // AlignmentA: 128 bits / 4 bits = 32
    32                                    // AlignmentB: 128 bits / 4 bits = 32
>;

/// Query workspace size for INT4 GEMM.
size_t cutlass_gemm_i4_workspace_size(int M, int N, int K) {
    GemmI4::Arguments args(
        {M, N, K},
        {nullptr, K},
        {nullptr, K},
        {nullptr, N},
        {nullptr, N},
        {1.0f, 0.0f}
    );
    return GemmI4::get_workspace_size(args);
}

/// Launch INT4 x INT4 -> FP16 GEMM.
///
/// A: (M, K) row-major INT4 quantized activations (packed, 2 per byte)
/// B: (N, K) column-major INT4 quantized weights (packed, 2 per byte)
/// C: (M, N) row-major FP16 output
/// scale: dequantization scale applied in epilogue (alpha = scale_a * scale_b)
///
/// Returns 0 on success.
int cutlass_gemm_i4(
    const void *A, const void *B, void *C,
    int M, int N, int K,
    float scale, float beta,
    void *workspace, size_t workspace_size
) {
    GemmI4 gemm_op;

    GemmI4::Arguments args(
        {M, N, K},
        {static_cast<const cutlass::int4b_t*>(A), K},
        {static_cast<const cutlass::int4b_t*>(B), K},
        {static_cast<const cutlass::half_t*>(C), N},
        {static_cast<cutlass::half_t*>(C), N},
        {scale, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] i4 can_implement failed: %d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] i4 initialize failed: %d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] i4 run failed: %d\n", (int)status);
    }

    return (int)status;
}

} // extern "C"
