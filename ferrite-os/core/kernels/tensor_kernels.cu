/*
 * PTX-OS Tensor Kernels - Zero-Copy GPU Compute
 * Native CUDA kernel implementations for all tensor operations
 */

#include "gpu/tensor_ops.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// Kernel Configuration
// ============================================================================

#define PTX_BLOCK_SIZE 256
#define PTX_WARP_SIZE 32

// Calculate grid size for N elements
#define PTX_GRID_SIZE(n) (((n) + PTX_BLOCK_SIZE - 1) / PTX_BLOCK_SIZE)

// Math constants
#define PTX_SQRT_2_PI 2.5066282746310002f
#define PTX_SQRT_2_OVER_PI 0.7978845608028654f
#define PTX_GELU_COEF 0.044715f
#define PTX_SELU_ALPHA 1.6732632423543772f
#define PTX_SELU_LAMBDA 1.0507009873554805f

// ============================================================================
// Binary Operation Kernels (F32)
// ============================================================================

__global__ void k_add_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void k_sub_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void k_mul_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void k_div_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void k_max_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void k_min_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

__global__ void k_mod_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmodf(a[idx], b[idx]);
    }
}

// Scalar broadcast variants
__global__ void k_add_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void k_sub_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void k_mul_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void k_div_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

// ============================================================================
// Unary Operation Kernels (F32)
// ============================================================================

__global__ void k_neg_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -in[idx];
    }
}

__global__ void k_abs_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(in[idx]);
    }
}

__global__ void k_exp_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(in[idx]);
    }
}

__global__ void k_log_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = logf(in[idx]);
    }
}

__global__ void k_sqrt_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(in[idx]);
    }
}

__global__ void k_rsqrt_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = rsqrtf(in[idx]);
    }
}

__global__ void k_sin_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinf(in[idx]);
    }
}

__global__ void k_cos_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cosf(in[idx]);
    }
}

__global__ void k_tanh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(in[idx]);
    }
}

__global__ void k_ceil_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceilf(in[idx]);
    }
}

__global__ void k_floor_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floorf(in[idx]);
    }
}

__global__ void k_round_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = roundf(in[idx]);
    }
}

__global__ void k_log2_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log2f(in[idx]);
    }
}

__global__ void k_log10_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log10f(in[idx]);
    }
}

__global__ void k_tan_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanf(in[idx]);
    }
}

__global__ void k_sinh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinhf(in[idx]);
    }
}

__global__ void k_cosh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = coshf(in[idx]);
    }
}

__global__ void k_sign_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
}

__global__ void k_erf_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Abramowitz & Stegun approximation (max error ~1.5e-7)
        float x = in[idx];
        float sign = (x >= 0.0f) ? 1.0f : -1.0f;
        float a = fabsf(x);
        float t = 1.0f / (1.0f + 0.3275911f * a);
        float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * expf(-a * a);
        out[idx] = sign * y;
    }
}

__global__ void k_sqr_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x * x;
    }
}

__global__ void k_recip_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / in[idx];
    }
}

// ============================================================================
// Activation Kernels (F32)
// ============================================================================

__global__ void k_relu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

__global__ void k_relu6_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(6.0f, fmaxf(0.0f, in[idx]));
    }
}

__global__ void k_leaky_relu_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t n, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x > 0.0f ? x : alpha * x;
    }
}

__global__ void k_elu_f32(const float* __restrict__ in, float* __restrict__ out,
                          size_t n, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}

__global__ void k_selu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = PTX_SELU_LAMBDA * (x > 0.0f ? x : PTX_SELU_ALPHA * (expf(x) - 1.0f));
    }
}

__global__ void k_gelu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = PTX_SQRT_2_OVER_PI * (x + PTX_GELU_COEF * x3);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void k_sigmoid_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void k_silu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x / (1.0f + expf(-x));  // x * sigmoid(x)
    }
}

__global__ void k_softplus_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // Numerically stable: log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0
        if (x > 20.0f) {
            out[idx] = x;
        } else if (x < -20.0f) {
            out[idx] = expf(x);
        } else {
            out[idx] = log1pf(expf(x));
        }
    }
}

__global__ void k_mish_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float sp = (x > 20.0f) ? x : log1pf(expf(x));  // softplus
        out[idx] = x * tanhf(sp);
    }
}

__global__ void k_gelu_tanh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float inner = PTX_SQRT_2_OVER_PI * (x + PTX_GELU_COEF * x * x * x);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void k_hardswish_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    }
}

__global__ void k_hardsigmoid_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(fmaxf(in[idx] / 6.0f + 0.5f, 0.0f), 1.0f);
    }
}

// ============================================================================
// Affine/Transform Kernels (F32)
// ============================================================================

__global__ void k_affine_f32(const float* __restrict__ in, float* __restrict__ out,
                              size_t n, float mul, float add) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * mul + add;
    }
}

__global__ void k_clamp_f32(const float* __restrict__ in, float* __restrict__ out,
                             size_t n, float min_val, float max_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(max_val, fmaxf(min_val, in[idx]));
    }
}

__global__ void k_powf_f32(const float* __restrict__ in, float* __restrict__ out,
                            size_t n, float exp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(in[idx], exp);
    }
}

// ============================================================================
// Comparison Kernels (F32)
// ============================================================================

__global__ void k_cmp_eq_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_lt_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_le_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_gt_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_ge_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_ne_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// Where/Select Kernel
// ============================================================================

__global__ void k_where_f32(const uint8_t* __restrict__ cond, const float* __restrict__ t,
                             const float* __restrict__ f, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cond[idx] ? t[idx] : f[idx];
    }
}

// ============================================================================
// Copy/Fill Kernels
// ============================================================================

__global__ void k_copy_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void k_fill_f32(float* __restrict__ out, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

// ============================================================================
// Reduction Kernels (F32) - Using shared memory for efficiency
// ============================================================================

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Reduce along a dimension: [outer, reduce, inner] -> [outer, inner]
__global__ void k_reduce_sum_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    // Each thread sums multiple elements
    float sum = 0.0f;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        sum += in[in_idx];
    }

    // Warp-level reduction
    sum = warp_reduce_sum_f32(sum);

    // Store to shared memory
    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    // Final reduction in first warp
    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        sum = sdata[threadIdx.x];
    } else {
        sum = 0.0f;
    }

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) {
            out[out_idx] = sum;
        }
    }
}

__global__ void k_reduce_max_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    float max_val = -INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        max_val = fmaxf(max_val, in[in_idx]);
    }

    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        max_val = sdata[threadIdx.x];
    } else {
        max_val = -INFINITY;
    }

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) {
            out[out_idx] = max_val;
        }
    }
}

__global__ void k_reduce_min_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    float min_val = INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        min_val = fminf(min_val, in[in_idx]);
    }

    min_val = warp_reduce_min_f32(min_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = min_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        min_val = sdata[threadIdx.x];
    } else {
        min_val = INFINITY;
    }

    if (wid == 0) {
        min_val = warp_reduce_min_f32(min_val);
        if (lane == 0) {
            out[out_idx] = min_val;
        }
    }
}

__device__ __forceinline__ float warp_reduce_prod_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void k_reduce_prod_f32(const float* __restrict__ in, float* __restrict__ out,
                                   size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    float prod = 1.0f;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        prod *= in[in_idx];
    }

    prod = warp_reduce_prod_f32(prod);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = prod;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        prod = sdata[threadIdx.x];
    } else {
        prod = 1.0f;
    }

    if (wid == 0) {
        prod = warp_reduce_prod_f32(prod);
        if (lane == 0) {
            out[out_idx] = prod;
        }
    }
}

// Argmax/Argmin: one thread per (outer, inner) lane, sequential scan over reduce dim
__global__ void k_reduce_argmax_f32(const float* __restrict__ in, int32_t* __restrict__ out,
                                     size_t outer, size_t reduce, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * inner;
    if (lane >= total) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * reduce * inner + i;

    float best = -INFINITY;
    int32_t best_idx = 0;
    for (size_t r = 0; r < reduce; r++) {
        float v = in[base + r * inner];
        if (v > best) {
            best = v;
            best_idx = (int32_t)r;
        }
    }
    out[lane] = best_idx;
}

__global__ void k_reduce_argmin_f32(const float* __restrict__ in, int32_t* __restrict__ out,
                                     size_t outer, size_t reduce, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * inner;
    if (lane >= total) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * reduce * inner + i;

    float best = INFINITY;
    int32_t best_idx = 0;
    for (size_t r = 0; r < reduce; r++) {
        float v = in[base + r * inner];
        if (v < best) {
            best = v;
            best_idx = (int32_t)r;
        }
    }
    out[lane] = best_idx;
}

// ============================================================================
// Softmax Kernels (F32)
// ============================================================================

// Softmax along last dimension: [batch, dim] -> [batch, dim]
__global__ void k_softmax_f32(const float* __restrict__ in, float* __restrict__ out,
                               size_t batch, size_t dim) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const float* in_row = in + batch_idx * dim;
    float* out_row = out + batch_idx * dim;

    // Step 1: Find max for numerical stability
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        max_val = sdata[threadIdx.x];
    } else {
        max_val = -INFINITY;
    }

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) {
            sdata[0] = max_val;  // Store final max
        }
    }
    __syncthreads();
    max_val = sdata[0];

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float exp_val = expf(in_row[i] - max_val);
        out_row[i] = exp_val;  // Store temporarily
        sum += exp_val;
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    if (threadIdx.x < numWarps) {
        sum = sdata[threadIdx.x];
    } else {
        sum = 0.0f;
    }

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) {
            sdata[0] = sum;  // Store final sum
        }
    }
    __syncthreads();
    sum = sdata[0];

    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}

__global__ void k_log_softmax_f32(const float* __restrict__ in, float* __restrict__ out,
                                   size_t batch, size_t dim) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const float* in_row = in + batch_idx * dim;
    float* out_row = out + batch_idx * dim;

    // Step 1: Find max
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        max_val = sdata[threadIdx.x];
    } else {
        max_val = -INFINITY;
    }

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) {
            sdata[0] = max_val;
        }
    }
    __syncthreads();
    max_val = sdata[0];

    // Step 2: Compute sum(exp(x - max))
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += expf(in_row[i] - max_val);
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    if (threadIdx.x < numWarps) {
        sum = sdata[threadIdx.x];
    } else {
        sum = 0.0f;
    }

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) {
            sdata[0] = sum;
        }
    }
    __syncthreads();
    sum = sdata[0];

    // Step 3: Compute x - max - log(sum)
    float log_sum = logf(sum);
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = in_row[i] - max_val - log_sum;
    }
}

// ============================================================================
// F64 Kernels
// ============================================================================

__global__ void k_add_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void k_sub_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

__global__ void k_mul_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void k_div_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

__global__ void k_neg_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -in[idx];
}

__global__ void k_exp_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = exp(in[idx]);
}

__global__ void k_log_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = log(in[idx]);
}

__global__ void k_sqrt_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrt(in[idx]);
}

__global__ void k_tanh_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanh(in[idx]);
}

__global__ void k_relu_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmax(0.0, in[idx]);
}

__global__ void k_gelu_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = in[idx];
        double x3 = x * x * x;
        double inner = 0.7978845608028654 * (x + 0.044715 * x3);
        out[idx] = 0.5 * x * (1.0 + tanh(inner));
    }
}

__global__ void k_sigmoid_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0 / (1.0 + exp(-in[idx]));
}

__global__ void k_affine_f64(const double* __restrict__ in, double* __restrict__ out,
                              size_t n, double mul, double add) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * mul + add;
}

// F64 Reduction
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void k_reduce_sum_f64(const double* __restrict__ in, double* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ double sdata_f64[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    double sum = 0.0;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        sum += in[in_idx];
    }

    sum = warp_reduce_sum_f64(sum);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = sum;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    sum = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : 0.0;

    if (wid == 0) {
        sum = warp_reduce_sum_f64(sum);
        if (lane == 0) out[out_idx] = sum;
    }
}

__global__ void k_reduce_max_f64(const double* __restrict__ in, double* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ double sdata_f64[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    double max_val = -INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        max_val = fmax(max_val, in[in_idx]);
    }

    max_val = warp_reduce_max_f64(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = max_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    max_val = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : -INFINITY;

    if (wid == 0) {
        max_val = warp_reduce_max_f64(max_val);
        if (lane == 0) out[out_idx] = max_val;
    }
}

__global__ void k_reduce_min_f64(const double* __restrict__ in, double* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ double sdata_f64[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    double min_val = INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        min_val = fmin(min_val, in[in_idx]);
    }

    min_val = warp_reduce_max_f64(min_val);  // Note: using max for clarity in code

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = min_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    min_val = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : INFINITY;

    if (wid == 0) {
        for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
            min_val = fmin(min_val, __shfl_down_sync(0xffffffff, min_val, offset));
        }
        if (lane == 0) out[out_idx] = min_val;
    }
}

__global__ void k_softmax_f64(const double* __restrict__ in, double* __restrict__ out,
                               size_t batch, size_t dim) {
    extern __shared__ double sdata_f64[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const double* in_row = in + batch_idx * dim;
    double* out_row = out + batch_idx * dim;

    // Find max
    double max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmax(max_val, in_row[i]);
    }
    max_val = warp_reduce_max_f64(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = max_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    max_val = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : -INFINITY;

    if (wid == 0) {
        max_val = warp_reduce_max_f64(max_val);
        if (lane == 0) sdata_f64[0] = max_val;
    }
    __syncthreads();
    max_val = sdata_f64[0];

    // Compute exp and sum
    double sum = 0.0;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        double exp_val = exp(in_row[i] - max_val);
        out_row[i] = exp_val;
        sum += exp_val;
    }
    sum = warp_reduce_sum_f64(sum);

    if (lane == 0) sdata_f64[wid] = sum;
    __syncthreads();

    sum = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : 0.0;

    if (wid == 0) {
        sum = warp_reduce_sum_f64(sum);
        if (lane == 0) sdata_f64[0] = sum;
    }
    __syncthreads();
    sum = sdata_f64[0];

    // Normalize
    double inv_sum = 1.0 / sum;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}

// ============================================================================
// F16 Kernels (using CUDA half)
// ============================================================================

__global__ void k_add_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void k_sub_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void k_mul_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void k_div_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hdiv(a[idx], b[idx]);
    }
}

__global__ void k_relu_f16(const __half* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half zero = __float2half(0.0f);
        out[idx] = __hgt(in[idx], zero) ? in[idx] : zero;
    }
}

__global__ void k_gelu_f16(const __half* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(in[idx]);
        float x3 = x * x * x;
        float inner = PTX_SQRT_2_OVER_PI * (x + PTX_GELU_COEF * x3);
        float result = 0.5f * x * (1.0f + tanhf(inner));
        out[idx] = __float2half(result);
    }
}

__global__ void k_sigmoid_f16(const __half* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(in[idx]);
        out[idx] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

// F16 Softmax
__global__ void k_softmax_f16(const __half* __restrict__ in, __half* __restrict__ out,
                               size_t batch, size_t dim) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const __half* in_row = in + batch_idx * dim;
    __half* out_row = out + batch_idx * dim;

    // Find max (in float for stability)
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, __half2float(in_row[i]));
    }
    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata[wid] = max_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    max_val = (threadIdx.x < numWarps) ? sdata[threadIdx.x] : -INFINITY;

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) sdata[0] = max_val;
    }
    __syncthreads();
    max_val = sdata[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float exp_val = expf(__half2float(in_row[i]) - max_val);
        out_row[i] = __float2half(exp_val);
        sum += exp_val;
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) sdata[wid] = sum;
    __syncthreads();

    sum = (threadIdx.x < numWarps) ? sdata[threadIdx.x] : 0.0f;

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) sdata[0] = sum;
    }
    __syncthreads();
    sum = sdata[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(out_row[i]) * inv_sum;
        out_row[i] = __float2half(val);
    }
}

// Cast kernels
__global__ void k_cast_f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void k_cast_f16_to_f32(const __half* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// ============================================================================
// External API Implementation
// ============================================================================

// F32 Binary
extern "C" void ptx_tensor_add_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_add_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_sub_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_sub_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mul_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_mul_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_div_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_div_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_max_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_max_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_min_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_min_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mod_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_mod_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// F32 Scalar broadcast
extern "C" void ptx_tensor_add_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_add_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

extern "C" void ptx_tensor_sub_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_sub_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

extern "C" void ptx_tensor_mul_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_mul_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

extern "C" void ptx_tensor_div_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_div_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

// F32 Unary
extern "C" void ptx_tensor_neg_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_neg_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_abs_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_abs_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_exp_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_exp_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_log_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sqrt_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sqrt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_rsqrt_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_rsqrt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sin_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sin_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cos_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_cos_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_tanh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_tanh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_ceil_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_ceil_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_floor_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_floor_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_round_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_round_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sqr_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sqr_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_recip_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_recip_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log2_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_log2_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log10_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_log10_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_tan_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_tan_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sinh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sinh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cosh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_cosh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sign_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sign_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_erf_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_erf_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

// F32 Activations
extern "C" void ptx_tensor_relu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_relu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_relu6_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_relu6_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_leaky_relu_f32(float* in, float* out, size_t n, float alpha, cudaStream_t stream) {
    k_leaky_relu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, alpha);
}

extern "C" void ptx_tensor_elu_f32(float* in, float* out, size_t n, float alpha, cudaStream_t stream) {
    k_elu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, alpha);
}

extern "C" void ptx_tensor_selu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_selu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_gelu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sigmoid_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sigmoid_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_silu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_silu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_softplus_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_softplus_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_mish_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_mish_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_tanh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_gelu_tanh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_hardswish_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_hardswish_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_hardsigmoid_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_hardsigmoid_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

// F32 Affine/Transform
extern "C" void ptx_tensor_affine_f32(float* in, float* out, size_t n, float mul, float add, cudaStream_t stream) {
    k_affine_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, mul, add);
}

extern "C" void ptx_tensor_clamp_f32(float* in, float* out, size_t n, float min_val, float max_val, cudaStream_t stream) {
    k_clamp_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, min_val, max_val);
}

extern "C" void ptx_tensor_powf_f32(float* in, float* out, size_t n, float exp, cudaStream_t stream) {
    k_powf_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, exp);
}

// F32 Comparison
extern "C" void ptx_tensor_cmp_eq_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_eq_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_lt_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_lt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_le_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_le_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_gt_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_gt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_ge_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_ge_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_ne_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_ne_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// F32 Where
extern "C" void ptx_tensor_where_f32(uint8_t* cond, float* t, float* f, float* out, size_t n, cudaStream_t stream) {
    k_where_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(cond, t, f, out, n);
}

// F32 Copy/Fill
extern "C" void ptx_tensor_copy_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_copy_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_fill_f32(float* out, size_t n, float value, cudaStream_t stream) {
    k_fill_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, value);
}

// F32 Reductions
extern "C" void ptx_tensor_reduce_sum_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_sum_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_mean_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    // Mean = sum / count, so we compute sum first then scale
    ptx_tensor_reduce_sum_f32(in, out, outer, reduce, inner, stream);
    float scale = 1.0f / (float)reduce;
    k_mul_scalar_f32<<<PTX_GRID_SIZE(outer * inner), PTX_BLOCK_SIZE, 0, stream>>>(out, scale, out, outer * inner);
}

extern "C" void ptx_tensor_reduce_max_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_max_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_min_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_min_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_prod_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_prod_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_argmax_f32(float* in, int32_t* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t total = outer * inner;
    if (total == 0) return;
    k_reduce_argmax_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_argmin_f32(float* in, int32_t* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t total = outer * inner;
    if (total == 0) return;
    k_reduce_argmin_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, reduce, inner);
}

// F32 Gather (index along middle axis)
__global__ void k_gather_f32(const float* in, const int32_t* indices, float* out,
                             size_t outer, size_t input_dim_size, size_t idx_dim_size, size_t inner) {
    size_t total = outer * idx_dim_size * inner;
    for (unsigned int elem = blockIdx.x * blockDim.x + threadIdx.x; elem < total; elem += blockDim.x * gridDim.x) {
        size_t o = elem / (idx_dim_size * inner);
        size_t k = (elem / inner) % idx_dim_size;
        size_t i = elem % inner;

        int32_t idx = indices[o * idx_dim_size * inner + k * inner + i];
        if (idx < 0) idx += (int32_t)input_dim_size;
        if (idx < 0) idx = 0;
        if (idx >= (int32_t)input_dim_size) idx = (int32_t)input_dim_size - 1;

        out[elem] = in[o * input_dim_size * inner + (size_t)idx * inner + i];
    }
}

extern "C" void ptx_tensor_gather_f32(float* in, int32_t* indices, float* out,
                                       size_t outer, size_t input_dim_size, size_t idx_dim_size, size_t inner,
                                       cudaStream_t stream) {
    size_t total = outer * idx_dim_size * inner;
    if (total == 0) return;
    k_gather_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(in, indices, out, outer, input_dim_size, idx_dim_size, inner);
}

// F32 CumSum (inclusive prefix sum along middle axis)
// Tensor is [outer, dim_size, inner]; each thread handles one (outer, inner) lane.
__global__ void k_cumsum_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;

    float acc = 0.0f;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        acc += in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cumsum_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cumsum_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// Cumulative product
__global__ void k_cumprod_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;
    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;
    float acc = 1.0f;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        acc *= in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cumprod_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cumprod_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// Cumulative max
__global__ void k_cummax_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;
    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;
    float acc = -INFINITY;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        if (in[idx] > acc) acc = in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cummax_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cummax_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// Cumulative min
__global__ void k_cummin_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;
    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;
    float acc = INFINITY;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        if (in[idx] < acc) acc = in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cummin_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cummin_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// ============================================================================
// Cast kernels: f32 <-> i32
// ============================================================================

__global__ void k_cast_f32_to_i32(const float* __restrict__ in, int* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (int)in[idx]; }
}

__global__ void k_cast_i32_to_f32(const int* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (float)in[idx]; }
}

extern "C" void ptx_tensor_cast_f32_to_i32(float* in, int* out, size_t n, cudaStream_t stream) {
    k_cast_f32_to_i32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cast_i32_to_f32(int* in, float* out, size_t n, cudaStream_t stream) {
    k_cast_i32_to_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

// ============================================================================
// Random number generation (xorshift128+ PRNG)
// ============================================================================

__global__ void k_rand_f32(float* __restrict__ out, size_t n, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Per-thread state from seed + idx
        unsigned long long s0 = seed ^ (idx * 6364136223846793005ULL + 1442695040888963407ULL);
        unsigned long long s1 = s0 * 6364136223846793005ULL + 1442695040888963407ULL;
        // xorshift128+
        s1 ^= s0;
        s0 = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
        s1 = s1 << 36 | s1 >> 28;
        unsigned long long result = s0 + s1;
        // Convert to float in [0, 1)
        out[idx] = (float)(result >> 40) / 16777216.0f;
    }
}

__global__ void k_randn_f32(float* __restrict__ out, size_t n, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Generate two uniform values using different seeds
        unsigned long long s0 = seed ^ (idx * 6364136223846793005ULL + 1442695040888963407ULL);
        unsigned long long s1 = s0 * 6364136223846793005ULL + 1442695040888963407ULL;
        s1 ^= s0;
        s0 = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
        s1 = s1 << 36 | s1 >> 28;
        float u1 = (float)((s0 + s1) >> 40) / 16777216.0f;
        // Second round
        s1 ^= s0;
        s0 = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
        s1 = s1 << 36 | s1 >> 28;
        float u2 = (float)((s0 + s1) >> 40) / 16777216.0f;
        // Box-Muller transform
        u1 = fmaxf(u1, 1e-7f); // avoid log(0)
        out[idx] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
}

extern "C" void ptx_tensor_rand_f32(float* out, size_t n, unsigned long long seed, cudaStream_t stream) {
    k_rand_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, seed);
}

extern "C" void ptx_tensor_randn_f32(float* out, size_t n, unsigned long long seed, cudaStream_t stream) {
    k_randn_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, seed);
}

// F32 TopK selection along middle axis
// Tensor is [outer, dim_size, inner]; each thread handles one (outer, inner) lane,
// selecting k extreme values via repeated scan.
__global__ void k_topk_f32(const float* in, float* values_out, int32_t* indices_out,
                           size_t outer, size_t dim_size, size_t inner, size_t k, int largest) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t in_base  = o * dim_size * inner + i;
    size_t out_base = o * k * inner + i;
    size_t actual_k = k < dim_size ? k : dim_size;

    for (size_t ki = 0; ki < actual_k; ki++) {
        float best_val;
        int32_t best_idx = -1;
        for (size_t d = 0; d < dim_size; d++) {
            float val = in[in_base + d * inner];
            bool skip = false;
            for (size_t prev = 0; prev < ki; prev++) {
                if (indices_out[out_base + prev * inner] == (int32_t)d) { skip = true; break; }
            }
            if (skip) continue;
            if (best_idx < 0) {
                best_val = val; best_idx = (int32_t)d;
            } else {
                bool better = largest ? (val > best_val) : (val < best_val);
                bool tie = (val == best_val && (int32_t)d < best_idx);
                if (better || tie) { best_val = val; best_idx = (int32_t)d; }
            }
        }
        values_out [out_base + ki * inner] = best_val;
        if (indices_out) indices_out[out_base + ki * inner] = best_idx;
    }
}

extern "C" void ptx_tensor_topk_f32(float* in, float* values_out, int32_t* indices_out,
                                     size_t outer, size_t dim_size, size_t inner,
                                     size_t k, int largest, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0 || k == 0) return;
    k_topk_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(
        in, values_out, indices_out, outer, dim_size, inner, k, largest);
}

// F32 Index-Select (select slices along a dimension using a 1D index array)
// Input shape: [left_size, src_dim_size, right_size]
// Output shape: [left_size, ids_dim_size, right_size]
// out[l][k][r] = input[l][ids[k]][r]
__global__ void k_index_select_f32(const float* input, const int32_t* ids, float* output,
                                    size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size) {
    size_t total = left_size * ids_dim_size * right_size;
    for (unsigned int elem = blockIdx.x * blockDim.x + threadIdx.x; elem < total; elem += blockDim.x * gridDim.x) {
        size_t l = elem / (ids_dim_size * right_size);
        size_t k = (elem / right_size) % ids_dim_size;
        size_t r = elem % right_size;

        int32_t idx = ids[k];
        if (idx < 0) idx += (int32_t)src_dim_size;
        if (idx < 0) idx = 0;
        if (idx >= (int32_t)src_dim_size) idx = (int32_t)src_dim_size - 1;

        output[elem] = input[l * src_dim_size * right_size + (size_t)idx * right_size + r];
    }
}

extern "C" void ptx_tensor_index_select_f32(float* input, int32_t* ids, float* output,
                                              size_t left_size, size_t src_dim_size, size_t ids_dim_size,
                                              size_t right_size, cudaStream_t stream) {
    size_t total = left_size * ids_dim_size * right_size;
    if (total == 0) return;
    k_index_select_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        input, ids, output, left_size, src_dim_size, ids_dim_size, right_size);
}

// F32 Scatter-Add (atomically accumulate src values into output at index positions)
// src shape: [left_size, src_dim_size, right_size]
// output shape: [left_size, dst_dim_size, right_size] (must be zero-initialized)
// output[l][ids[k]][r] += src[l][k][r]
__global__ void k_scatter_add_f32(const int32_t* ids, const float* src, float* output,
                                   size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size) {
    size_t total = left_size * src_dim_size * right_size;
    for (unsigned int elem = blockIdx.x * blockDim.x + threadIdx.x; elem < total; elem += blockDim.x * gridDim.x) {
        size_t l = elem / (src_dim_size * right_size);
        size_t k = (elem / right_size) % src_dim_size;
        size_t r = elem % right_size;

        int32_t idx = ids[k];
        if (idx < 0) idx += (int32_t)dst_dim_size;
        if (idx >= 0 && idx < (int32_t)dst_dim_size) {
            atomicAdd(&output[l * dst_dim_size * right_size + (size_t)idx * right_size + r], src[elem]);
        }
    }
}

extern "C" void ptx_tensor_scatter_add_f32(int32_t* ids, float* src, float* output,
                                             size_t left_size, size_t src_dim_size, size_t dst_dim_size,
                                             size_t right_size, cudaStream_t stream) {
    size_t total = left_size * src_dim_size * right_size;
    if (total == 0) return;
    k_scatter_add_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        ids, src, output, left_size, src_dim_size, dst_dim_size, right_size);
}

// F32 Argsort (bitonic sort network — one block per row, shared memory)
// Input: [nrows, ncols], Output: [nrows, ncols] of uint32 indices
// ncols_pad must be the next power of 2 >= ncols.
__global__ void k_argsort_asc_f32(const float* input, unsigned int* output,
                                   int ncols, int ncols_pad) {
    int row = blockIdx.x;
    const float* x = input + row * ncols;
    extern __shared__ int shmem[];

    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        shmem[col] = col;
    }
    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (shmem[col] >= ncols ||
                            (shmem[ixj] < ncols && x[shmem[col]] > x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    } else {
                        if (shmem[ixj] >= ncols ||
                            (shmem[col] < ncols && x[shmem[col]] < x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        output[row * ncols + col] = (unsigned int)shmem[col];
    }
}

__global__ void k_argsort_desc_f32(const float* input, unsigned int* output,
                                    int ncols, int ncols_pad) {
    int row = blockIdx.x;
    const float* x = input + row * ncols;
    extern __shared__ int shmem[];

    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        shmem[col] = col;
    }
    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (shmem[col] >= ncols ||
                            (shmem[ixj] < ncols && x[shmem[col]] < x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    } else {
                        if (shmem[ixj] >= ncols ||
                            (shmem[col] < ncols && x[shmem[col]] > x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        output[row * ncols + col] = (unsigned int)shmem[col];
    }
}

extern "C" void ptx_tensor_argsort_f32(float* input, unsigned int* output,
                                         size_t nrows, size_t ncols, int ascending,
                                         cudaStream_t stream) {
    if (nrows == 0 || ncols == 0) return;
    // Round ncols up to next power of 2
    int ncols_pad = 1;
    while (ncols_pad < (int)ncols) ncols_pad *= 2;
    int block = (ncols_pad < PTX_BLOCK_SIZE) ? ncols_pad : PTX_BLOCK_SIZE;
    size_t smem = ncols_pad * sizeof(int);
    if (ascending) {
        k_argsort_asc_f32<<<(int)nrows, block, smem, stream>>>(input, output, (int)ncols, ncols_pad);
    } else {
        k_argsort_desc_f32<<<(int)nrows, block, smem, stream>>>(input, output, (int)ncols, ncols_pad);
    }
}

// F32 Softmax
extern "C" void ptx_tensor_softmax_f32(float* in, float* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(float);
    k_softmax_f32<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

extern "C" void ptx_tensor_log_softmax_f32(float* in, float* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(float);
    k_log_softmax_f32<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

// F64 API
extern "C" void ptx_tensor_add_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_add_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_sub_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_sub_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mul_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_mul_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_div_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_div_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_neg_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_neg_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_exp_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_exp_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_log_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sqrt_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_sqrt_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_tanh_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_tanh_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_relu_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_relu_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_gelu_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sigmoid_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_sigmoid_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_reduce_sum_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(double);
    k_reduce_sum_f64<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_max_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(double);
    k_reduce_max_f64<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_min_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(double);
    k_reduce_min_f64<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_softmax_f64(double* in, double* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(double);
    k_softmax_f64<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

extern "C" void ptx_tensor_affine_f64(double* in, double* out, size_t n, double mul, double add, cudaStream_t stream) {
    k_affine_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, mul, add);
}

// F16 API
extern "C" void ptx_tensor_add_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_add_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_sub_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_sub_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mul_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_mul_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_div_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_div_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_relu_f16(__half* in, __half* out, size_t n, cudaStream_t stream) {
    k_relu_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_f16(__half* in, __half* out, size_t n, cudaStream_t stream) {
    k_gelu_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sigmoid_f16(__half* in, __half* out, size_t n, cudaStream_t stream) {
    k_sigmoid_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_softmax_f16(__half* in, __half* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(float);
    k_softmax_f16<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

extern "C" void ptx_tensor_cast_f32_to_f16(float* in, __half* out, size_t n, cudaStream_t stream) {
    k_cast_f32_to_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cast_f16_to_f32(__half* in, float* out, size_t n, cudaStream_t stream) {
    k_cast_f16_to_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

// ============================================================================
// Arange / Linspace kernels
// ============================================================================

__global__ void k_arange_f32(float* __restrict__ out, size_t n, float start, float step) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + (float)idx * step;
    }
}

extern "C" void ptx_tensor_arange_f32(float* out, size_t n, float start, float step, cudaStream_t stream) {
    k_arange_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, start, step);
}

// ============================================================================
// Logical ops (on U8 boolean tensors)
// ============================================================================

__global__ void k_logical_and_u8(const unsigned char* __restrict__ a, const unsigned char* __restrict__ b,
                                  unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] && b[idx]) ? 1 : 0; }
}

__global__ void k_logical_or_u8(const unsigned char* __restrict__ a, const unsigned char* __restrict__ b,
                                 unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] || b[idx]) ? 1 : 0; }
}

__global__ void k_logical_not_u8(const unsigned char* __restrict__ in, unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = in[idx] ? 0 : 1; }
}

__global__ void k_logical_xor_u8(const unsigned char* __restrict__ a, const unsigned char* __restrict__ b,
                                  unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] != b[idx]) ? 1 : 0; }
}

extern "C" void ptx_tensor_logical_and_u8(unsigned char* a, unsigned char* b, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_and_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
extern "C" void ptx_tensor_logical_or_u8(unsigned char* a, unsigned char* b, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_or_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
extern "C" void ptx_tensor_logical_not_u8(unsigned char* in, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_not_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}
extern "C" void ptx_tensor_logical_xor_u8(unsigned char* a, unsigned char* b, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_xor_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// ============================================================================
// Broadcast binary op kernel
// ============================================================================

// Computes flat index from nd-index using strides
__device__ size_t broadcast_offset(size_t flat_idx, const size_t* out_shape, const size_t* in_strides, int ndim) {
    size_t offset = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = flat_idx % out_shape[d];
        flat_idx /= out_shape[d];
        offset += coord * in_strides[d];
    }
    return offset;
}

__global__ void k_broadcast_binary_f32(const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ out, size_t n,
                                        const size_t* out_shape, const size_t* a_strides,
                                        const size_t* b_strides, int ndim, int op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = a[broadcast_offset(idx, out_shape, a_strides, ndim)];
        float vb = b[broadcast_offset(idx, out_shape, b_strides, ndim)];
        float result;
        switch (op) {
            case 0: result = va + vb; break;
            case 1: result = va - vb; break;
            case 2: result = va * vb; break;
            case 3: result = va / vb; break;
            case 4: result = fmaxf(va, vb); break;
            case 5: result = fminf(va, vb); break;
            case 6: result = fmodf(va, vb); break;
            default: result = 0.0f;
        }
        out[idx] = result;
    }
}

extern "C" void ptx_tensor_broadcast_binary_f32(
    float* a, float* b, float* out, size_t n,
    size_t* out_shape, size_t* a_strides, size_t* b_strides,
    int ndim, int op, cudaStream_t stream
) {
    k_broadcast_binary_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        a, b, out, n, out_shape, a_strides, b_strides, ndim, op);
}

// ============================================================================
// Pad kernel (constant padding)
// ============================================================================

// 2D padding: pad [N, C, H, W] with constant value
__global__ void k_pad2d_f32(const float* __restrict__ in, float* __restrict__ out,
                             int N, int C, int H, int W,
                             int pad_top, int pad_bottom, int pad_left, int pad_right,
                             float pad_value) {
    int oH = H + pad_top + pad_bottom;
    int oW = W + pad_left + pad_right;
    size_t total = (size_t)N * (size_t)C * (size_t)oH * (size_t)oW;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int ow = idx % oW;
        int oh = (idx / oW) % oH;
        int c  = (idx / (oW * oH)) % C;
        int n  = idx / (oW * oH * C);

        int ih = oh - pad_top;
        int iw = ow - pad_left;

        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            out[idx] = in[((n * C + c) * H + ih) * W + iw];
        } else {
            out[idx] = pad_value;
        }
    }
}

extern "C" void ptx_tensor_pad2d_f32(
    float* in, float* out,
    int N, int C, int H, int W,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    float pad_value, cudaStream_t stream
) {
    int oH = H + pad_top + pad_bottom;
    int oW = W + pad_left + pad_right;
    size_t total = (size_t)N * C * oH * oW;
    k_pad2d_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, pad_value);
}

// ============================================================================
// Repeat / Tile kernel
// ============================================================================

__global__ void k_repeat_f32(const float* __restrict__ in, float* __restrict__ out,
                              size_t out_n, const size_t* in_shape, const size_t* out_shape,
                              int ndim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_n) {
        // Map output index to input index by taking modulo of each dim
        size_t in_idx = 0;
        size_t remaining = idx;
        size_t in_stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            size_t coord = remaining % out_shape[d];
            remaining /= out_shape[d];
            size_t in_coord = coord % in_shape[d];
            in_idx += in_coord * in_stride;
            in_stride *= in_shape[d];
        }
        out[idx] = in[in_idx];
    }
}

extern "C" void ptx_tensor_repeat_f32(
    float* in, float* out, size_t out_n,
    size_t* in_shape, size_t* out_shape, int ndim, cudaStream_t stream
) {
    k_repeat_f32<<<PTX_GRID_SIZE(out_n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, out_n, in_shape, out_shape, ndim);
}

// ============================================================================
// Masked fill kernel
// ============================================================================

__global__ void k_masked_fill_f32(float* __restrict__ data, const unsigned char* __restrict__ mask,
                                   float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx]) {
            data[idx] = value;
        }
    }
}

extern "C" void ptx_tensor_masked_fill_f32(float* data, unsigned char* mask, float value, size_t n, cudaStream_t stream) {
    k_masked_fill_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(data, mask, value, n);
}

// ============================================================================
// Generic Dispatch Implementation
// ============================================================================

extern "C" cudaError_t ptx_tensor_dispatch(const PTXTensorOp* op) {
    if (!op) return cudaErrorInvalidValue;

    cudaStream_t stream = op->stream;

    switch (op->opcode) {
        // Binary ops
        case PTX_OP_ADD:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_add_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_add_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_add_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_SUB:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_sub_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_sub_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_sub_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_MUL:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_mul_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_mul_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_mul_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_DIV:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_div_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_div_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_div_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        // Unary ops
        case PTX_OP_NEG:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_neg_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_neg_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_EXP:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_exp_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_exp_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_LOG:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_log_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_log_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SQRT:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sqrt_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_sqrt_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_TANH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_tanh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_tanh_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SIN:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sin_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_COS:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_cos_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ABS:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_abs_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SQR:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sqr_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_RECIP:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_recip_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_CEIL:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_ceil_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_FLOOR:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_floor_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ROUND:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_round_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SIGN:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sign_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_LOG2:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_log2_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_LOG10:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_log10_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_TAN:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_tan_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SINH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sinh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_COSH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_cosh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ERF:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_erf_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_MOD:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_mod_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            break;

        // Activations
        case PTX_OP_RELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_relu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_relu_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F16) ptx_tensor_relu_f16((__half*)op->input_a, (__half*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_GELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_gelu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_gelu_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F16) ptx_tensor_gelu_f16((__half*)op->input_a, (__half*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SIGMOID:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sigmoid_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_sigmoid_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F16) ptx_tensor_sigmoid_f16((__half*)op->input_a, (__half*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SILU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_silu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_elu_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, stream);
            break;

        case PTX_OP_LEAKY_RELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_leaky_relu_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, stream);
            break;

        case PTX_OP_SELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_selu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SOFTPLUS:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_softplus_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_MISH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_mish_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_GELU_TANH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_gelu_tanh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_HARDSWISH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_hardswish_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_HARDSIGMOID:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_hardsigmoid_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        // Reductions
        case PTX_OP_REDUCE_SUM:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_sum_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_reduce_sum_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_MEAN:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_mean_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_MAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_max_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_reduce_max_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_MIN:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_min_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_reduce_min_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_PROD:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_prod_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_ARGMAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_argmax_f32((float*)op->input_a, (int32_t*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_ARGMIN:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_argmin_f32((float*)op->input_a, (int32_t*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        // Softmax
        case PTX_OP_SOFTMAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_softmax_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_softmax_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_softmax_f16((__half*)op->input_a, (__half*)op->output, op->outer_size, op->reduce_size, stream);
            }
            break;

        case PTX_OP_LOG_SOFTMAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_log_softmax_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, stream);
            }
            break;

        // Affine/Transform
        case PTX_OP_AFFINE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_affine_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, op->scalar_b, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_affine_f64((double*)op->input_a, (double*)op->output, op->elem_count, (double)op->scalar_a, (double)op->scalar_b, stream);
            }
            break;

        case PTX_OP_CLAMP:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_clamp_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, op->scalar_b, stream);
            }
            break;

        case PTX_OP_POW:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_powf_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, stream);
            }
            break;

        // Comparison
        case PTX_OP_CMP_EQ:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_eq_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_LT:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_lt_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_LE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_le_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_GT:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_gt_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_GE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_ge_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_NE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_ne_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        // Where
        case PTX_OP_WHERE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_where_f32((uint8_t*)op->input_a, (float*)op->input_b, (float*)op->output, (float*)op->output, op->elem_count, stream);
            }
            break;

        // Copy
        case PTX_OP_COPY:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_copy_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_FILL:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_fill_f32((float*)op->output, op->elem_count, op->scalar_a, stream);
            }
            break;

        case PTX_OP_GATHER:
            // For gather dispatch: outer_size=outer, reduce_size=input_dim_size,
            // inner_size=idx_dim_size, elem_count=inner
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_gather_f32((float*)op->input_a, (int32_t*)op->input_b, (float*)op->output,
                                      op->outer_size, op->reduce_size, op->inner_size, op->elem_count, stream);
            }
            break;

        case PTX_OP_CUMSUM:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cumsum_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_TOPK:
            // For topk dispatch: outer_size=outer, reduce_size=dim_size, inner_size=inner,
            // elem_count=k, scalar_a=largest (1.0 or 0.0), input_b=indices output (or NULL)
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_topk_f32((float*)op->input_a, (float*)op->output, (int32_t*)op->input_b,
                                    op->outer_size, op->reduce_size, op->inner_size,
                                    op->elem_count, (int)(op->scalar_a > 0.5f), stream);
            }
            break;

        case PTX_OP_INDEX_SELECT:
            // outer_size=left, reduce_size=src_dim, inner_size=ids_dim, elem_count=right
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_index_select_f32((float*)op->input_a, (int32_t*)op->input_b, (float*)op->output,
                                            op->outer_size, op->reduce_size, op->inner_size, op->elem_count, stream);
            }
            break;

        case PTX_OP_SCATTER_ADD:
            // outer_size=left, reduce_size=src_dim, inner_size=dst_dim, elem_count=right
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_scatter_add_f32((int32_t*)op->input_b, (float*)op->input_a, (float*)op->output,
                                           op->outer_size, op->reduce_size, op->inner_size, op->elem_count, stream);
            }
            break;

        case PTX_OP_ARGSORT:
            // outer_size=nrows, reduce_size=ncols, scalar_a=ascending (1.0 or 0.0)
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_argsort_f32((float*)op->input_a, (unsigned int*)op->output,
                                       op->outer_size, op->reduce_size,
                                       (int)(op->scalar_a > 0.5f), stream);
            }
            break;

        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

extern "C" cudaError_t ptx_tensor_dispatch_sync(const PTXTensorOp* op) {
    cudaError_t err = ptx_tensor_dispatch(op);
    if (err != cudaSuccess) return err;

    if (op->stream) {
        return cudaStreamSynchronize(op->stream);
    } else {
        return cudaDeviceSynchronize();
    }
}

// ============================================================================
// CUDA Graph Integration
// ============================================================================

extern "C" cudaError_t ptx_tensor_graph_begin_capture(cudaStream_t stream) {
    return cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
}

extern "C" cudaError_t ptx_tensor_graph_end_capture(cudaStream_t stream, cudaGraph_t* graph_out) {
    return cudaStreamEndCapture(stream, graph_out);
}

extern "C" cudaError_t ptx_tensor_graph_instantiate(cudaGraph_t graph, cudaGraphExec_t* exec_out) {
    return cudaGraphInstantiate(exec_out, graph, NULL, NULL, 0);
}

extern "C" cudaError_t ptx_tensor_graph_launch(cudaGraphExec_t exec, cudaStream_t stream) {
    return cudaGraphLaunch(exec, stream);
}

// ============================================================================
// Strided Copy kernel (for making non-contiguous tensors contiguous)
// ============================================================================

__global__ void k_strided_copy_f32(const float* __restrict__ in, float* __restrict__ out,
                                    const size_t* __restrict__ shape,
                                    const size_t* __restrict__ in_strides,
                                    int ndim, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Decompose linear output index into nd-index, then compute source offset
    size_t src_offset = 0;
    size_t remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        src_offset += coord * in_strides[d];
    }
    out[idx] = in[src_offset];
}

extern "C" void ptx_tensor_strided_copy_f32(
    const float* in, float* out,
    const size_t* shape, const size_t* in_strides,
    int ndim, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    k_strided_copy_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, shape, in_strides, ndim, n);
}

__global__ void k_strided_copy_f64(const double* __restrict__ in, double* __restrict__ out,
                                    const size_t* __restrict__ shape,
                                    const size_t* __restrict__ in_strides,
                                    int ndim, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t src_offset = 0;
    size_t remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        src_offset += coord * in_strides[d];
    }
    out[idx] = in[src_offset];
}

extern "C" void ptx_tensor_strided_copy_f64(
    const double* in, double* out,
    const size_t* shape, const size_t* in_strides,
    int ndim, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    k_strided_copy_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, shape, in_strides, ndim, n);
}

__global__ void k_strided_copy_u8(const unsigned char* __restrict__ in, unsigned char* __restrict__ out,
                                   const size_t* __restrict__ shape,
                                   const size_t* __restrict__ in_strides,
                                   int ndim, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t src_offset = 0;
    size_t remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        src_offset += coord * in_strides[d];
    }
    out[idx] = in[src_offset];
}

extern "C" void ptx_tensor_strided_copy_u8(
    const unsigned char* in, unsigned char* out,
    const size_t* shape, const size_t* in_strides,
    int ndim, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    k_strided_copy_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, shape, in_strides, ndim, n);
}

// ============================================================================
// Im2col / Col2im (for Conv2d)
// ============================================================================

__global__ void k_im2col_f32(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * kH * kW * H_out * W_out;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t w_out = idx % W_out;
        size_t tmp = idx / W_out;
        size_t h_out = tmp % H_out;
        tmp /= H_out;
        size_t kw = tmp % kW;
        tmp /= kW;
        size_t kh = tmp % kH;
        tmp /= kH;
        size_t c = tmp % C;
        size_t n = tmp / C;

        int h_in = (int)h_out * strideH - padH + (int)kh * dilationH;
        int w_in = (int)w_out * strideW - padW + (int)kw * dilationW;

        float val = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            val = data_im[((n * C + c) * H + h_in) * W + w_in];
        }

        // Output layout: (N, C*kH*kW, H_out*W_out)
        size_t col_c = c * kH * kW + kh * kW + kw;
        size_t col_hw = h_out * W_out + w_out;
        data_col[(n * (C * kH * kW) + col_c) * (H_out * W_out) + col_hw] = val;
    }
}

extern "C" void ptx_tensor_im2col_f32(
    const float* data_im, float* data_col,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * kH * kW * H_out * W_out;
    if (total == 0) return;
    k_im2col_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        data_im, data_col, N, C, H, W, kH, kW,
        padH, padW, strideH, strideW, dilationH, dilationW, H_out, W_out);
}

__global__ void k_col2im_f32(
    const float* __restrict__ data_col,
    float* __restrict__ data_im,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * H * W;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t w = idx % W;
        size_t tmp = idx / W;
        size_t h = tmp % H;
        tmp /= H;
        size_t c = tmp % C;
        size_t n = tmp / C;

        float sum = 0.0f;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int h_padded = (int)h + padH - kh * dilationH;
                int w_padded = (int)w + padW - kw * dilationW;

                if (h_padded % strideH != 0 || w_padded % strideW != 0) continue;
                int ho = h_padded / strideH;
                int wo = w_padded / strideW;
                if (ho < 0 || ho >= H_out || wo < 0 || wo >= W_out) continue;

                size_t col_c = c * kH * kW + kh * kW + kw;
                size_t col_hw = ho * W_out + wo;
                sum += data_col[(n * (C * kH * kW) + col_c) * (H_out * W_out) + col_hw];
            }
        }
        data_im[idx] = sum;
    }
}

extern "C" void ptx_tensor_col2im_f32(
    const float* data_col, float* data_im,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * H * W;
    if (total == 0) return;
    // Zero the output first
    cudaMemsetAsync(data_im, 0, total * sizeof(float), stream);
    k_col2im_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        data_col, data_im, N, C, H, W, kH, kW,
        padH, padW, strideH, strideW, dilationH, dilationW, H_out, W_out);
}

// ============================================================================
// Pooling kernels
// ============================================================================

__global__ void k_max_pool2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * H_out * W_out;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t wo = idx % W_out;
        size_t tmp = idx / W_out;
        size_t ho = tmp % H_out;
        tmp /= H_out;
        size_t c = tmp % C;
        size_t n = tmp / C;

        float max_val = -INFINITY;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int hi = (int)ho * strideH - padH + kh;
                int wi = (int)wo * strideW - padW + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    float v = input[((n * C + c) * H + hi) * W + wi];
                    if (v > max_val) max_val = v;
                }
            }
        }
        output[idx] = max_val;
    }
}

extern "C" void ptx_tensor_max_pool2d_f32(
    const float* input, float* output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * H_out * W_out;
    if (total == 0) return;
    k_max_pool2d_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        input, output, N, C, H, W, kH, kW, strideH, strideW, padH, padW, H_out, W_out);
}

__global__ void k_avg_pool2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * H_out * W_out;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t wo = idx % W_out;
        size_t tmp = idx / W_out;
        size_t ho = tmp % H_out;
        tmp /= H_out;
        size_t c = tmp % C;
        size_t n = tmp / C;

        float sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int hi = (int)ho * strideH - padH + kh;
                int wi = (int)wo * strideW - padW + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    sum += input[((n * C + c) * H + hi) * W + wi];
                    count++;
                }
            }
        }
        output[idx] = (count > 0) ? sum / (float)count : 0.0f;
    }
}

extern "C" void ptx_tensor_avg_pool2d_f32(
    const float* input, float* output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * H_out * W_out;
    if (total == 0) return;
    k_avg_pool2d_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        input, output, N, C, H, W, kH, kW, strideH, strideW, padH, padW, H_out, W_out);
}

// ============================================================================
// Optimizer kernels
// ============================================================================

__global__ void k_sgd_step_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ velocity,
    size_t n,
    float lr, float momentum, float weight_decay
) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float g = grads[idx];
        if (weight_decay != 0.0f) {
            g += weight_decay * params[idx];
        }
        if (momentum != 0.0f && velocity != NULL) {
            velocity[idx] = momentum * velocity[idx] + g;
            g = velocity[idx];
        }
        params[idx] -= lr * g;
    }
}

extern "C" void ptx_tensor_sgd_step_f32(
    float* params, const float* grads, float* velocity,
    size_t n, float lr, float momentum, float weight_decay,
    cudaStream_t stream
) {
    if (n == 0) return;
    k_sgd_step_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        params, grads, velocity, n, lr, momentum, weight_decay);
}

__global__ void k_adam_step_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    size_t n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2
) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float g = grads[idx];
        if (weight_decay != 0.0f) {
            g += weight_decay * params[idx];
        }
        // Update biased first moment
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // Update biased second moment
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        // Bias-corrected estimates
        float m_hat = m[idx] / bc1;
        float v_hat = v[idx] / bc2;
        // Update parameters
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" void ptx_tensor_adam_step_f32(
    float* params, const float* grads, float* m, float* v,
    size_t n, float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2,
    cudaStream_t stream
) {
    if (n == 0) return;
    k_adam_step_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        params, grads, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}
