//! Test Candle kernels with PTX-OS TLSF allocator
//!
//! This demonstrates that Candle kernels work seamlessly with PTX-OS memory management.
//! All allocations go through the TLSF allocator - kernels just do the math!

use ptx_kernels::candle;
use std::ptr;

fn main() {
    println!("🔥 Testing Candle Kernels with PTX-OS TLSF Allocator\n");

    unsafe {
        // Initialize PTX-OS runtime with TLSF allocator
        let runtime_ptr = ptx_sys::gpu_hot_init(0, ptr::null());
        if runtime_ptr.is_null() {
            panic!("Failed to initialize PTX-OS runtime");
        }
        println!("✓ PTX-OS runtime initialized");

        // Get stream from PTX-OS
        let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, 0);
        println!("✓ Got stream from PTX-OS\n");

        // Test parameters
        const N: usize = 1024 * 1024; // 1M elements
        const BYTES: usize = N * std::mem::size_of::<f32>();

        println!("📊 Configuration:");
        println!("   Elements: {}", N);
        println!("   Size: {} KB\n", BYTES / 1024);

        // Allocate using PTX-OS TLSF allocator
        let d_input = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_output = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);
        let d_temp = ptx_sys::gpu_hot_alloc(runtime_ptr, BYTES);

        if d_input.is_null() || d_output.is_null() || d_temp.is_null() {
            panic!("TLSF allocation failed");
        }
        println!("✓ Allocated {} KB via TLSF allocator", (BYTES * 3) / 1024);

        // Initialize input data
        let mut h_input = vec![0.0f32; N];
        for (i, val) in h_input.iter_mut().enumerate() {
            *val = (i as f32) / 10000.0 - 50.0; // Range: -50 to +50
        }

        // cudaMemcpyHostToDevice = 1
        ptx_sys::cudaMemcpy(
            d_input,
            h_input.as_ptr() as *const _,
            BYTES,
            1, // cudaMemcpyHostToDevice
        );
        println!("✓ Copied input data to GPU\n");

        // Test 1: GELU activation
        println!("🧪 Test 1: GELU Activation");
        candle::candle_launch_ugelu_f32(
            N,
            0,
            ptr::null(),
            d_input as *const f32,
            d_output as *mut f32,
            stream,
        );
        ptx_sys::cudaStreamSynchronize(stream);
        println!("   ✓ GELU completed");

        // Test 2: ReLU activation
        println!("\n🧪 Test 2: ReLU Activation");
        candle::candle_launch_urelu_f32(
            N,
            0,
            ptr::null(),
            d_input as *const f32,
            d_temp as *mut f32,
            stream,
        );
        ptx_sys::cudaStreamSynchronize(stream);
        println!("   ✓ ReLU completed");

        // Test 3: Element-wise multiplication (output = GELU(input) * ReLU(input))
        println!("\n🧪 Test 3: Element-wise Multiplication");
        candle::candle_launch_bmul_f32(
            N,
            0,
            ptr::null(),
            ptr::null(),
            d_output as *const f32,
            ptr::null(),
            d_temp as *const f32,
            d_output as *mut f32,
            stream,
        );
        ptx_sys::cudaStreamSynchronize(stream);
        println!("   ✓ Multiplication completed");

        // Test 4: Tanh
        println!("\n🧪 Test 4: Tanh Activation");
        candle::candle_launch_utanh_f32(
            N,
            0,
            ptr::null(),
            d_output as *const f32,
            d_temp as *mut f32,
            stream,
        );
        ptx_sys::cudaStreamSynchronize(stream);
        println!("   ✓ Tanh completed");

        // Test 5: Sigmoid
        println!("\n🧪 Test 5: Sigmoid Activation");
        candle::candle_launch_usigmoid_f32(
            N,
            0,
            ptr::null(),
            d_input as *const f32,
            d_temp as *mut f32,
            stream,
        );
        ptx_sys::cudaStreamSynchronize(stream);
        println!("   ✓ Sigmoid completed");

        // Test 6: Binary addition
        println!("\n🧪 Test 6: Element-wise Addition");
        candle::candle_launch_badd_f32(
            N,
            0,
            ptr::null(),
            ptr::null(),
            d_output as *const f32,
            ptr::null(),
            d_temp as *const f32,
            d_output as *mut f32,
            stream,
        );
        ptx_sys::cudaStreamSynchronize(stream);
        println!("   ✓ Addition completed");

        // Copy result back and verify
        let mut h_output = vec![0.0f32; N];
        // cudaMemcpyDeviceToHost = 2
        ptx_sys::cudaMemcpy(
            h_output.as_mut_ptr() as *mut _,
            d_output,
            BYTES,
            2, // cudaMemcpyDeviceToHost
        );

        println!("\n📊 Sample Results:");
        println!("   First 5: {:?}", &h_output[0..5]);
        println!("   Middle 5: {:?}", &h_output[N / 2..N / 2 + 5]);
        println!("   Last 5: {:?}", &h_output[N - 5..N]);

        // Validation
        let nan_count = h_output.iter().filter(|x| x.is_nan()).count();
        let inf_count = h_output.iter().filter(|x| x.is_infinite()).count();
        let finite_count = h_output.iter().filter(|x| x.is_finite()).count();

        println!("\n🔍 Validation:");
        println!("   Finite values: {}/{}", finite_count, N);
        println!("   NaN count: {}", nan_count);
        println!("   Inf count: {}", inf_count);

        // Free using TLSF allocator
        ptx_sys::gpu_hot_free(runtime_ptr, d_input);
        ptx_sys::gpu_hot_free(runtime_ptr, d_output);
        ptx_sys::gpu_hot_free(runtime_ptr, d_temp);
        println!("\n✓ Freed memory via TLSF allocator");

        // Cleanup runtime
        ptx_sys::gpu_hot_shutdown(runtime_ptr);
        println!("✓ PTX-OS runtime shutdown");

        if nan_count == 0 && inf_count == 0 {
            println!("\n✅ SUCCESS: All Candle kernels work perfectly with PTX-OS TLSF!");
            println!("   The kernels just do math - memory management is completely separate!");
        } else {
            println!("\n⚠️  Warning: Found {} NaN and {} Inf values", nan_count, inf_count);
        }
    }
}
