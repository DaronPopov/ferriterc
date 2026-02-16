use std::env;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use aten_ptx::{init_pytorch_tlsf_ex, set_torch_stream, sync_all_streams};
use ptx_runtime::{PTXStableConfig, PTX_STABLE_ABI_VERSION, PtxRuntime};
use tch::{kind::Kind, no_grad, Device, Tensor};

#[derive(Clone)]
struct Config {
    duration_seconds: u64,
    iterations: usize,
    batch_size: i64,
    input_dim: i64,
    hidden_dim: i64,
    streams: u32,
    pool_fraction: f32,
    report_every: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            duration_seconds: 60,
            iterations: 0,
            batch_size: 32,
            input_dim: 256,
            hidden_dim: 256,
            streams: 8,
            pool_fraction: 0.20,
            report_every: 10,
        }
    }
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--duration" => {
                cfg.duration_seconds = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --duration"))?
                    .parse()?;
            }
            "--iterations" => {
                cfg.iterations = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --iterations"))?
                    .parse()?;
            }
            "--batch-size" => {
                cfg.batch_size = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --batch-size"))?
                    .parse()?;
            }
            "--input-dim" => {
                cfg.input_dim = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --input-dim"))?
                    .parse()?;
            }
            "--hidden-dim" => {
                cfg.hidden_dim = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --hidden-dim"))?
                    .parse()?;
            }
            "--streams" => {
                cfg.streams = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --streams"))?
                    .parse()?;
            }
            "--pool" => {
                cfg.pool_fraction = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --pool"))?
                    .parse()?;
            }
            "--report" => {
                cfg.report_every = args
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --report"))?
                    .parse()?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(anyhow!("unknown argument: {arg}")),
        }
    }

    if cfg.streams == 0 {
        cfg.streams = 1;
    }
    cfg.pool_fraction = cfg.pool_fraction.clamp(0.01, 0.9);
    if cfg.report_every == 0 {
        cfg.report_every = 10;
    }

    Ok(cfg)
}

fn print_help() {
    println!("Usage: streaming_tokenizer.rs [options]");
    println!();
    println!("  --duration N     Seconds to stream (ignored if --iterations > 0) [default: 60]");
    println!("  --iterations N   Number of iterations (overrides duration when >0)");
    println!("  --batch-size N   Input batch size [default: 32]");
    println!("  --input-dim N    Synthetic input dimensionality [default: 256]");
    println!("  --hidden-dim N   Hidden layer size [default: 256]");
    println!("  --streams N      PTX stream count [default: 8]");
    println!("  --pool F         TLSF pool fraction (0.01-0.9) [default: 0.20]");
    println!("  --report N       Show status every N iterations [default: 10]");
    println!("  -h, --help       Print this help");
}

fn describe_summary(cfg: &Config, iterations: usize, tokens: i64, start: Instant) {
    let elapsed = start.elapsed();
    println!();
    println!("Streaming summary:");
    println!("  iterations:     {}", iterations);
    println!("  duration:       {:.2}s", elapsed.as_secs_f64());
    println!("  batch size:     {}", cfg.batch_size);
    println!("  input dim:      {}", cfg.input_dim);
    println!("  hidden dim:     {}", cfg.hidden_dim);
    println!("  stream count:   {}", cfg.streams);
    println!("  pool fraction:  {:.2}", cfg.pool_fraction);
    println!("  tokens total:   {}", tokens);
    println!("  sample throughput: {:.1} tokens/s", tokens as f64 / elapsed.as_secs_f64());
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: cfg.pool_fraction,
        fixed_pool_size: 0,
        reserve_vram: 0,
        max_streams: cfg.streams,
        quiet_init: 0,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };

    let runtime = PtxRuntime::with_stable_config(0, Some(runtime_cfg))?;
    runtime.export_for_hook();
    runtime.export_context();
    init_pytorch_tlsf_ex(0, cfg.pool_fraction as f64, cfg.streams)
        .map_err(|e| anyhow!(e))?;

    let device = Device::Cuda(0);
    let kernel1 = Tensor::randn(&[cfg.input_dim, cfg.hidden_dim], (Kind::Float, device));
    let bias1 = Tensor::zeros(&[cfg.hidden_dim], (Kind::Float, device));
    let kernel2 = Tensor::randn(&[cfg.hidden_dim, cfg.hidden_dim], (Kind::Float, device));
    let bias2 = Tensor::zeros(&[cfg.hidden_dim], (Kind::Float, device));

    println!("Streaming workload starting (Jetson-friendly) ...");
    let start = Instant::now();
    let mut iteration = 0usize;
    let mut tokens_processed = 0i64;

    let duration = Duration::from_secs(cfg.duration_seconds.max(1));
    while (cfg.iterations == 0 && start.elapsed() < duration)
        || (cfg.iterations > 0 && iteration < cfg.iterations)
    {
        let stream_id = iteration % cfg.streams as usize;
        set_torch_stream(stream_id);

        let inputs = Tensor::randn(&[cfg.batch_size, cfg.input_dim], (Kind::Float, device));
        let logits = no_grad(|| {
            let hidden = inputs.matmul(&kernel1) + &bias1;
            let activated = hidden.relu();
            let mid = activated.matmul(&kernel2) + &bias2;
            mid.softmax(-1, Kind::Float)
        });

        let _token_sum = logits.sum(Kind::Float);
        tokens_processed += cfg.batch_size;

        sync_all_streams();

        if iteration % cfg.report_every == 0 {
            let sample_rate = cfg.batch_size as f64 * (iteration as f64 + 1.0)
                / start.elapsed().as_secs_f64().max(1.0);
            println!(
                "iteration {} stream {} sample rate {:.1} samples/s",
                iteration + 1,
                stream_id,
                sample_rate
            );
        }

        iteration += 1;
    }

    let _ = runtime.sync_all();
    describe_summary(&cfg, iteration, tokens_processed, start);
    Ok(())
}
