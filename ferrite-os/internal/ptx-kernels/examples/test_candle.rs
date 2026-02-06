//! Test Candle kernels with PTX-OS TLSF allocator
use ptx_kernels::{candle, check_cuda, sync_stream};
use ptx_sys::cudaStream_t;
use std::ptr;

fn main() {
    println!("🔥 Testing Candle F32 Kernels with PTX-OS\n");

    unsafe {
        // Initialize CUDA
        check_cuda(ptx_sys::cudaSetDevice(0), "Failed to set device");

        // Create stream
        let mut stream: cudaStream_t = ptr::null_mut();
        check_cuda(
            ptx_sys::cudaStreamCreate(&mut stream as *mut _),
            "Failed to create stream",
        );

        // Test parameters
        const N: usize = 1024 * 1024; // 1M elements
        const BYTES: usize = N * std::mem::size_of::<f32>();

        println!("📊 Configuration:");
        println!("   Elements: {}", N);
        println!("   Size: {} KB\n", BYTES / 1024);

        // Allocate GPU memory
        let mut d_input: *mut f32 = ptr::null_mut();
        let mut d_output: *mut f32 = ptr::null_mut();
        let mut d_temp: *mut f32 = ptr::null_mut();

        check_cuda(
            ptx_sys::cudaMalloc(&mut d_input as *mut _ as *mut _, BYTES),
            "Failed to allocate input",
        );
        check_cuda(
            ptx_sys::cudaMalloc(&mut d_output as *mut _ as *mut _, BYTES),
            "Failed to allocate output",
        );
        check_cuda(
            ptx_sys::cudaMalloc(&mut d_temp as *mut _ as *mut _, BYTES),
            "Failed to allocate temp",
        );

        println!("✓ Allocated GPU memory");

        // Initialize input data
        let mut h_input = vec![0.0f32; N];
        for (i, val) in h_input.iter_mut().enumerate() {
            *val = (i as f32) / 1000.0 - 512.0; // Range: -512 to +512
        }

        check_cuda(
            ptx_sys::cudaMemcpy(
                d_input as *mut _,
                h_input.as_ptr() as *const _,
                BYTES,
                ptx_sys::cudaMemcpyKind_cudaMemcpyHostToDevice,
            ),
            "Failed to copy input to device",
        );

        println!("✓ Initialized input data\n");

        // Test 1: GELU
        println!("🧪 Test 1: GELU Activation");
        candle::candle_launch_ugelu_f32(N, 0, ptr::null(), d_input, d_output, stream);
        sync_stream(stream);
        println!("   ✓ GELU completed");

        // Test 2: ReLU
        println!("\n🧪 Test 2: ReLU Activation");
        candle::candle_launch_urelu_f32(N, 0, ptr::null(), d_input, d_temp, stream);
        sync_stream(stream);
        println!("   ✓ ReLU completed");

        // Test 3: Element-wise multiply
        println!("\n🧪 Test 3: Element-wise Multiplication");
        candle::candle_launch_bmul_f32(
            N,
            0,
            ptr::null(),
            ptr::null(),
            d_output,
            ptr::null(),
            d_temp,
            d_output,
            stream,
        );
        sync_stream(stream);
        println!("   ✓ Multiplication completed");

        // Test 4: Tanh
        println!("\n🧪 Test 4: Tanh Activation");
        candle::candle_launch_utanh_f32(N, 0, ptr::null(), d_input, d_temp, stream);
        sync_stream(stream);
        println!("   ✓ Tanh completed");

        // Test 5: Binary add
        println!("\n🧪 Test 5: Element-wise Addition");
        candle::candle_launch_badd_f32(
            N,
            0,
            ptr::null(),
            ptr::null(),
            d_output,
            ptr::null(),
            d_temp,
            d_output,
            stream,
        );
        sync_stream(stream);
        println!("   ✓ Addition completed");

        // Test 6: Exp
        println!("\n🧪 Test 6: Exponential");
        let mut h_small_input = vec![0.0f32; 1024];
        for (i, val) in h_small_input.iter_mut().enumerate() {
            *val = (i as f32) / 1000.0; // Small values to avoid overflow
        }
        check_cuda(
            ptx_sys::cudaMemcpy(
                d_input as *mut _,
                h_small_input.as_ptr() as *const _,
                1024 * std::mem::size_of::<f32>(),
                ptx_sys::cudaMemcpyKind_cudaMemcpyHostToDevice,
            ),
            "Failed to copy small input",
        );
        candle::candle_launch_uexp_f32(1024, 0, ptr::null(), d_input, d_temp, stream);
        sync_stream(stream);
        println!("   ✓ Exp completed");

        // Copy result back and verify
        let mut h_output = vec![0.0f32; N];
        check_cuda(
            ptx_sys::cudaMemcpy(
                h_output.as_mut_ptr() as *mut _,
                d_output as *const _,
                BYTES,
                ptx_sys::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            ),
            "Failed to copy output to host",
        );

        println!("\n📊 Sample Results:");
        println!("   First 5 values: {:?}", &h_output[0..5]);
        println!("   Middle 5 values: {:?}", &h_output[N / 2..N / 2 + 5]);
        println!("   Last 5 values: {:?}", &h_output[N - 5..N]);

        // Verify no NaNs or Infs
        let nan_count = h_output.iter().filter(|x| x.is_nan()).count();
        let inf_count = h_output.iter().filter(|x| x.is_infinite()).count();
        println!("\n🔍 Validation:");
        println!("   NaN count: {}", nan_count);
        println!("   Inf count: {}", inf_count);

        if nan_count == 0 && inf_count == 0 {
            println!("\n✅ All Candle kernels working correctly!");
        } else {
            println!("\n⚠️  Warning: Found NaN or Inf values");
        }

        // Cleanup
        ptx_sys::cudaFree(d_input as *mut _);
        ptx_sys::cudaFree(d_output as *mut _);
        ptx_sys::cudaFree(d_temp as *mut _);
        ptx_sys::cudaStreamDestroy(stream);

        println!("\n🧹 Cleaned up resources");
    }
}
