use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use aten_ptx::{init_pytorch_tlsf_ex, num_streams, set_torch_stream, sync_all_streams, reset_torch_stream};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{Device, Kind, Tensor};

const DEFAULT_STREAMS: u32 = 32;
const DEFAULT_WAVE: usize = 8;
const DEFAULT_MICRO_BATCH: i64 = 8;
const DEFAULT_HIDDEN: i64 = 2048;
const DEFAULT_LORA_RANK: i64 = 16;
const DEFAULT_LORA_ALPHA: f64 = 32.0;
const DEFAULT_TEXT_KEY: &str = "text";

#[derive(Clone)]
struct Config {
    weights_dir: Option<PathBuf>,
    adapter_path: Option<PathBuf>,
    dataset: PathBuf,
    dataset_format: DatasetFormat,
    text_key: String,
    shard_mb: usize,
    streams: u32,
    wave_streams: usize,
    micro_batch: i64,
    hidden: i64,
    lora_rank: i64,
    lora_alpha: f64,
    max_samples: usize,
    split_ratio: f64,
}

#[derive(Clone, Copy)]
enum DatasetFormat {
    Auto,
    Txt,
    Jsonl,
}

impl DatasetFormat {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "txt" => Ok(Self::Txt),
            "jsonl" => Ok(Self::Jsonl),
            _ => Err(anyhow::anyhow!("invalid format: {s}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Txt => "txt",
            Self::Jsonl => "jsonl",
        }
    }
}

fn detect_format(path: &PathBuf) -> DatasetFormat {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ext == "jsonl" { DatasetFormat::Jsonl } else { DatasetFormat::Txt }
}

fn extract_json_value(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let idx = line.find(&needle)?;
    let mut i = idx + needle.len();
    let bytes = line.as_bytes();
    while i < bytes.len() && bytes[i] != b':' { i += 1; }
    i += 1;
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() { i += 1; }
    if i >= bytes.len() || bytes[i] != b'"' { return None; }
    i += 1;
    let mut out = String::new();
    let mut esc = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        i += 1;
        if esc { out.push(c); esc = false; continue; }
        if c == '\\' { esc = true; continue; }
        if c == '"' { return Some(out); }
        out.push(c);
    }
    None
}

fn load_samples(path: &PathBuf, fmt: DatasetFormat, key: &str) -> Result<Vec<String>> {
    let fmt = match fmt {
        DatasetFormat::Auto => detect_format(path),
        x => x,
    };
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        match fmt {
            DatasetFormat::Txt => samples.push(trimmed.to_string()),
            DatasetFormat::Jsonl => {
                if let Some(text) = extract_json_value(trimmed, key) {
                    let t = text.trim();
                    if !t.is_empty() { samples.push(t.to_string()); }
                }
            }
            DatasetFormat::Auto => {}
        }
    }
    Ok(samples)
}

fn encode_text(text: &str, len: i64, seed: u64) -> Tensor {
    let mut out = vec![0f32; len as usize];
    let bytes = text.as_bytes();
    if bytes.is_empty() { return Tensor::from_slice(&out).view([len]); }
    let mut h = 1469598103934665603u64 ^ seed;
    for (i, &b) in bytes.iter().enumerate() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
        let idx = (h as usize + i) % out.len();
        out[idx] += ((b as f32 / 255.0) * 2.0) - 1.0;
    }
    let n = (bytes.len() as f32).sqrt().max(1.0);
    for v in &mut out { *v /= n; }
    Tensor::from_slice(&out).view([len])
}

struct EvalMetrics {
    total_loss: f64,
    sample_count: usize,
    non_finite: usize,
    per_shard_loss: Vec<f64>,
    per_shard_count: Vec<usize>,
}

impl EvalMetrics {
    fn new(shard_count: usize) -> Self {
        Self {
            total_loss: 0.0,
            sample_count: 0,
            non_finite: 0,
            per_shard_loss: vec![0.0; shard_count],
            per_shard_count: vec![0; shard_count],
        }
    }

    fn record(&mut self, shard_idx: usize, loss: f64) {
        if loss.is_finite() {
            self.total_loss += loss;
            self.sample_count += 1;
            if shard_idx < self.per_shard_loss.len() {
                self.per_shard_loss[shard_idx] += loss;
                self.per_shard_count[shard_idx] += 1;
            }
        } else {
            self.non_finite += 1;
        }
    }

    fn avg_loss(&self) -> f64 {
        if self.sample_count > 0 { self.total_loss / self.sample_count as f64 } else { f64::NAN }
    }

    fn perplexity(&self) -> f64 {
        let avg = self.avg_loss();
        if avg.is_finite() { avg.exp() } else { f64::NAN }
    }
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config {
        weights_dir: None,
        adapter_path: None,
        dataset: PathBuf::new(),
        dataset_format: DatasetFormat::Auto,
        text_key: DEFAULT_TEXT_KEY.to_string(),
        shard_mb: 64,
        streams: DEFAULT_STREAMS,
        wave_streams: DEFAULT_WAVE,
        micro_batch: DEFAULT_MICRO_BATCH,
        hidden: DEFAULT_HIDDEN,
        lora_rank: DEFAULT_LORA_RANK,
        lora_alpha: DEFAULT_LORA_ALPHA,
        max_samples: 0,
        split_ratio: 1.0,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => cfg.dataset = PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing --dataset"))?),
            "--dataset-format" => cfg.dataset_format = DatasetFormat::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--text-key" => cfg.text_key = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?,
            "--weights-dir" => cfg.weights_dir = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)),
            "--adapter-path" => cfg.adapter_path = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)),
            "--shard-mb" => cfg.shard_mb = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--micro-batch" => cfg.micro_batch = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--hidden" => cfg.hidden = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-rank" => cfg.lora_rank = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--lora-alpha" => cfg.lora_alpha = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--max-samples" => cfg.max_samples = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--split-ratio" => cfg.split_ratio = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: validation_loop.rs --dataset PATH [options]");
                println!("  Runs a forward-only eval pass over a held-out dataset split.");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    if cfg.dataset.as_os_str().is_empty() {
        return Err(anyhow::anyhow!("--dataset is required"));
    }
    Ok(cfg)
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    aten_ptx::ensure_libtorch_cuda_loaded();
    if !tch::Cuda::is_available() {
        println!("CUDA not available, exiting");
        return Ok(());
    }

    // Init PTX runtime
    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.70,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: cfg.streams,
        quiet_init: 1,
        enable_leak_detection: 0,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = PtxRuntime::with_stable_config(0, Some(runtime_cfg))?;
    runtime.export_for_hook();
    runtime.export_context();
    init_pytorch_tlsf_ex(0, 0.70, cfg.streams).map_err(anyhow::Error::msg)?;

    let device = Device::Cuda(0);
    let active_streams = num_streams();
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f64;
    let inv_sqrt_d = 1.0 / (cfg.hidden as f64).sqrt();

    // Load and split dataset
    let all_samples = load_samples(&cfg.dataset, cfg.dataset_format, &cfg.text_key)?;
    let split_at = ((all_samples.len() as f64) * cfg.split_ratio) as usize;
    let eval_samples = if cfg.split_ratio < 1.0 {
        all_samples[split_at..].to_vec()
    } else {
        all_samples
    };

    let eval_count = if cfg.max_samples > 0 {
        eval_samples.len().min(cfg.max_samples)
    } else {
        eval_samples.len()
    };

    let shard_bytes = cfg.shard_mb * 1024 * 1024;
    let model_bytes = 100 * 1024 * 1024 * 1024usize; // placeholder
    let shards_per_pass = model_bytes.div_ceil(shard_bytes);

    println!("=== Ferrite Validation Loop ===\n");
    println!("  dataset:         {}", cfg.dataset.display());
    println!("  eval samples:    {}", eval_count);
    println!("  shards/pass:     {}", shards_per_pass);
    println!("  wave streams:    {}", cfg.wave_streams);
    println!("  micro batch:     {}", cfg.micro_batch);
    println!("  hidden:          {}", cfg.hidden);
    println!("  lora rank:       {}", cfg.lora_rank);

    let t_eval = Instant::now();
    let mut metrics = EvalMetrics::new(shards_per_pass);
    let mut sample_cursor = 0usize;

    tch::no_grad(|| -> Result<()> {
        let mut shard_base = 0usize;
        while shard_base < shards_per_pass {
            let wave = (shards_per_pass - shard_base).min(cfg.wave_streams);

            for local_idx in 0..wave {
                let shard_idx = shard_base + local_idx;
                let stream_id = shard_idx % active_streams;
                set_torch_stream(stream_id);

                let elems = (shard_bytes / std::mem::size_of::<f32>()) as i64;
                let rows = (elems / cfg.hidden).max(1);
                let used = rows * cfg.hidden;

                let base_w = (Tensor::randn([used], (Kind::Float, device)) * inv_sqrt_d)
                    .view([rows, cfg.hidden]);

                // In production, these would be loaded from adapter checkpoint
                let lora_a = Tensor::randn([rows, cfg.lora_rank], (Kind::Float, device));
                let lora_b = Tensor::randn([cfg.lora_rank, cfg.hidden], (Kind::Float, device));

                let delta = lora_a.matmul(&lora_b) * lora_scale;
                let effective_w = &base_w + &delta;

                // Build micro-batch from eval samples
                let mut rows_x = Vec::with_capacity(cfg.micro_batch as usize);
                for _ in 0..cfg.micro_batch {
                    let idx = sample_cursor % eval_count;
                    sample_cursor += 1;
                    rows_x.push(encode_text(&eval_samples[idx], cfg.hidden, 0xA11CE).view([1, cfg.hidden]));
                }
                let x = Tensor::cat(&rows_x, 0).to_device(device);

                let y = (x.matmul(&effective_w.transpose(0, 1)) * inv_sqrt_d).tanh();

                let mut rows_t = Vec::with_capacity(cfg.micro_batch as usize);
                for _ in 0..cfg.micro_batch {
                    let idx = sample_cursor % eval_count;
                    sample_cursor += 1;
                    rows_t.push(encode_text(&eval_samples[idx], rows, 0xBEEF).view([1, rows]));
                }
                let target = Tensor::cat(&rows_t, 0).to_device(device).tanh();

                let loss = (&y - &target).square().mean(Kind::Float);
                let lv = f64::try_from(&loss).unwrap_or(f64::NAN);
                metrics.record(shard_idx, lv);
            }

            sync_all_streams();
            reset_torch_stream();
            shard_base += wave;
        }
        Ok(())
    })?;

    tch::Cuda::synchronize(0);
    let wall = t_eval.elapsed().as_secs_f64();

    let s = runtime.tlsf_stats();

    println!("\n  Validation Results:");
    println!("    Avg Loss:       {:.6}", metrics.avg_loss());
    println!("    Perplexity:     {:.4}", metrics.perplexity());
    println!("    Samples:        {}", metrics.sample_count);
    println!("    Non-finite:     {}", metrics.non_finite);
    println!("    Wall time:      {:.2}s", wall);
    println!("    Peak VRAM:      {:.1} MB", s.peak_allocated as f64 / 1e6);
    println!("    Fragmentation:  {:.6}", s.fragmentation_ratio);

    println!("\nRESULT mode=validation");
    println!("RESULT avg_loss={:.9}", metrics.avg_loss());
    println!("RESULT perplexity={:.9}", metrics.perplexity());
    println!("RESULT eval_samples={}", metrics.sample_count);
    println!("RESULT non_finite={}", metrics.non_finite);
    println!("RESULT wall_s={:.6}", wall);
    println!("RESULT peak_vram_mb={:.6}", s.peak_allocated as f64 / 1e6);
    println!("RESULT fragmentation={:.9}", s.fragmentation_ratio);

    Ok(())
}
