/// JIT tiling & infix expressions — per-element math auto-parallelized on CUDA.
///
/// Demonstrates the new infix expression syntax and tile blocks:
///   - Scalar multiplication: `x * 2.0`
///   - Infix + builtin: `x * 2.0 + relu(x)`
///   - Tile block: `tile y over x: y = x * 2.0 + 1.0 end`
///   - Sub/div: `(x - 1.0) / 2.0`
///   - Parenthesized grouping: `(x + 1.0) * 2.0`

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{GpuLangRuntime, HostTensor};

fn main() {
    let mut jit = JitEngine::new();
    let runtime = GpuLangRuntime::new(0).expect("GPU runtime init");

    // ── Test 1: Scalar multiply ─────────────────────────────────
    {
        let script = r#"
            x = input([4])
            y = x * 2.0
            return y
        "#;
        let compiled = jit.compile(script).expect("compile scalar mul");
        let input = HostTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        println!("Test 1 — scalar multiply (x * 2.0):");
        println!("  input:  [1.0, 2.0, 3.0, 4.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[2.0, 4.0, 6.0, 8.0]);
        println!("  PASS");
    }

    // ── Test 2: Infix + builtin ─────────────────────────────────
    {
        let script = r#"
            x = input([4])
            y = x * 2.0 + relu(x)
            return y
        "#;
        let compiled = jit.compile(script).expect("compile infix+builtin");
        let input = HostTensor::new(vec![4], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // x*2.0: [-2.0, 0.0, 2.0, 4.0]
        // relu(x): [0.0, 0.0, 1.0, 2.0]
        // sum: [-2.0, 0.0, 3.0, 6.0]
        println!("\nTest 2 — infix + builtin (x * 2.0 + relu(x)):");
        println!("  input:  [-1.0, 0.0, 1.0, 2.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[-2.0, 0.0, 3.0, 6.0]);
        println!("  PASS");
    }

    // ── Test 3: Tile block ──────────────────────────────────────
    {
        let script = r#"
            x = input([4])
            tile y over x:
                y = x * 2.0 + 1.0
            end
            return y
        "#;
        let compiled = jit.compile(script).expect("compile tile");
        let input = HostTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // x*2.0: [2.0, 4.0, 6.0, 8.0]
        // +1.0:  [3.0, 5.0, 7.0, 9.0]
        println!("\nTest 3 — tile block (y = x * 2.0 + 1.0):");
        println!("  input:  [1.0, 2.0, 3.0, 4.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[3.0, 5.0, 7.0, 9.0]);
        println!("  PASS");
    }

    // ── Test 4: Sub and div ─────────────────────────────────────
    {
        let script = r#"
            x = input([4])
            y = (x - 1.0) / 2.0
            return y
        "#;
        let compiled = jit.compile(script).expect("compile sub/div");
        let input = HostTensor::new(vec![4], vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // (x-1.0): [0.0, 2.0, 4.0, 6.0]
        // /2.0:    [0.0, 1.0, 2.0, 3.0]
        println!("\nTest 4 — sub/div ((x - 1.0) / 2.0):");
        println!("  input:  [1.0, 3.0, 5.0, 7.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[0.0, 1.0, 2.0, 3.0]);
        println!("  PASS");
    }

    // ── Test 5: Parenthesized grouping ──────────────────────────
    {
        let script = r#"
            x = input([4])
            y = (x + 1.0) * 2.0
            return y
        "#;
        let compiled = jit.compile(script).expect("compile parens");
        let input = HostTensor::new(vec![4], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let output = runtime.execute(compiled, &[input]).unwrap();
        // (x+1.0): [1.0, 2.0, 3.0, 4.0]
        // *2.0:    [2.0, 4.0, 6.0, 8.0]
        println!("\nTest 5 — parenthesized ((x + 1.0) * 2.0):");
        println!("  input:  [0.0, 1.0, 2.0, 3.0]");
        println!("  output: {:?}", output.data());
        assert_eq!(output.data(), &[2.0, 4.0, 6.0, 8.0]);
        println!("  PASS");
    }

    println!("\nAll tiling/infix tests passed.");
}
