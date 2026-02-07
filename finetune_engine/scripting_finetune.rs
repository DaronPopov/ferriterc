use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use aten_ptx::{init_pytorch_tlsf_ex, num_streams, reset_torch_stream, set_torch_stream, sync_all_streams};
use ptx_runtime::{PTX_STABLE_ABI_VERSION, PTXStableConfig, PtxRuntime};
use tch::{Device, Kind, Tensor};

const DEFAULT_MODEL_GB: usize = 100;
const DEFAULT_SHARD_MB: usize = 64;
const DEFAULT_STEPS: usize = 200;
const DEFAULT_STREAMS: u32 = 64;
const DEFAULT_WAVE_STREAMS: usize = 16;
const DEFAULT_MICRO_BATCH: i64 = 8;
const DEFAULT_HIDDEN: i64 = 2048;
const DEFAULT_LORA_RANK: i64 = 16;
const DEFAULT_LORA_ALPHA: f64 = 32.0;
const DEFAULT_LR: f64 = 1e-3;
const DEFAULT_TEXT_KEY: &str = "text";

#[derive(Clone, Copy)]
enum WeightsSource {
    Synthetic,
    Directory,
}

impl WeightsSource {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "synthetic" => Ok(Self::Synthetic),
            "directory" => Ok(Self::Directory),
            _ => Err(anyhow::anyhow!("invalid --weights-source: {s} (expected synthetic|directory)")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::Directory => "directory",
        }
    }
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
            _ => Err(anyhow::anyhow!("invalid --dataset-format: {s} (expected auto|txt|jsonl)")),
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

#[derive(Clone)]
struct Config {
    weights_source: WeightsSource,
    weights_dir: Option<PathBuf>,
    model_gb: usize,
    shard_mb: usize,
    steps: usize,
    streams: u32,
    wave_streams: usize,
    micro_batch: i64,
    hidden: i64,
    lora_rank: i64,
    lora_alpha: f64,
    lr: f64,
    dataset: Option<PathBuf>,
    dataset_format: DatasetFormat,
    text_key: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            weights_source: WeightsSource::Synthetic,
            weights_dir: None,
            model_gb: DEFAULT_MODEL_GB,
            shard_mb: DEFAULT_SHARD_MB,
            steps: DEFAULT_STEPS,
            streams: DEFAULT_STREAMS,
            wave_streams: DEFAULT_WAVE_STREAMS,
            micro_batch: DEFAULT_MICRO_BATCH,
            hidden: DEFAULT_HIDDEN,
            lora_rank: DEFAULT_LORA_RANK,
            lora_alpha: DEFAULT_LORA_ALPHA,
            lr: DEFAULT_LR,
            dataset: None,
            dataset_format: DatasetFormat::Auto,
            text_key: DEFAULT_TEXT_KEY.to_string(),
        }
    }
}

#[derive(Clone)]
struct DatasetState {
    samples: Vec<String>,
    cursor: usize,
}

impl DatasetState {
    fn new(samples: Vec<String>) -> Self {
        Self { samples, cursor: 0 }
    }

    fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn next_text(&mut self) -> &str {
        let idx = self.cursor % self.samples.len();
        self.cursor = self.cursor.wrapping_add(1);
        &self.samples[idx]
    }
}

#[derive(Clone)]
struct Snap {
    allocs: u64,
    frees: u64,
    frag: f32,
    allocated: usize,
    peak: usize,
    total_pool: usize,
}

fn snap(runtime: &PtxRuntime) -> Snap {
    let s = runtime.tlsf_stats();
    Snap {
        allocs: s.total_allocations,
        frees: s.total_frees,
        frag: s.fragmentation_ratio,
        allocated: s.allocated_bytes,
        peak: s.peak_allocated,
        total_pool: s.total_pool_size,
    }
}

fn reset_peak(runtime: &PtxRuntime) {
    runtime.sync_all();
    runtime.poll_deferred(0);
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--weights-source" => {
                cfg.weights_source = WeightsSource::parse(
                    &args.next().ok_or_else(|| anyhow::anyhow!("missing value for --weights-source"))?,
                )?;
            }
            "--weights-dir" => {
                cfg.weights_dir = Some(PathBuf::from(
                    args.next().ok_or_else(|| anyhow::anyhow!("missing value for --weights-dir"))?,
                ));
            }
            "--model-gb" => cfg.model_gb = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --model-gb"))?.parse()?,
            "--shard-mb" => cfg.shard_mb = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --shard-mb"))?.parse()?,
            "--steps" => cfg.steps = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --steps"))?.parse()?,
            "--streams" => cfg.streams = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --streams"))?.parse()?,
            "--wave-streams" => cfg.wave_streams = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --wave-streams"))?.parse()?,
            "--micro-batch" => cfg.micro_batch = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --micro-batch"))?.parse()?,
            "--hidden" => cfg.hidden = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --hidden"))?.parse()?,
            "--lora-rank" => cfg.lora_rank = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --lora-rank"))?.parse()?,
            "--lora-alpha" => cfg.lora_alpha = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --lora-alpha"))?.parse()?,
            "--lr" => cfg.lr = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --lr"))?.parse()?,
            "--dataset" => {
                cfg.dataset = Some(PathBuf::from(
                    args.next().ok_or_else(|| anyhow::anyhow!("missing value for --dataset"))?,
                ));
            }
            "--dataset-format" => {
                cfg.dataset_format = DatasetFormat::parse(
                    &args.next().ok_or_else(|| anyhow::anyhow!("missing value for --dataset-format"))?,
                )?;
            }
            "--text-key" => {
                cfg.text_key = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --text-key"))?;
            }
            "-h" | "--help" => {
                println!("Usage: scripting_finetune.rs [options]");
                println!("  --weights-source synthetic|directory");
                println!("  --weights-dir PATH (required for directory source)");
                println!("  --model-gb N --shard-mb N --steps N");
                println!("  --streams N --wave-streams N");
                println!("  --micro-batch N --hidden N");
                println!("  --lora-rank N --lora-alpha F --lr F");
                println!("  --dataset PATH --dataset-format auto|txt|jsonl --text-key KEY");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    if cfg.model_gb == 0
        || cfg.shard_mb == 0
        || cfg.steps == 0
        || cfg.streams == 0
        || cfg.wave_streams == 0
        || cfg.micro_batch <= 0
        || cfg.hidden <= 0
        || cfg.lora_rank <= 0
        || cfg.lora_alpha <= 0.0
        || cfg.lr <= 0.0
    {
        return Err(anyhow::anyhow!("all numeric options must be > 0"));
    }

    if let Some(path) = cfg.weights_dir.clone() {
        cfg.weights_dir = Some(resolve_user_path(path));
    }

    if let Some(path) = cfg.dataset.clone() {
        cfg.dataset = Some(resolve_user_path(path));
    }

    if let WeightsSource::Directory = cfg.weights_source {
        let Some(dir) = &cfg.weights_dir else {
            return Err(anyhow::anyhow!("--weights-dir is required with --weights-source directory"));
        };
        if !dir.exists() {
            return Err(anyhow::anyhow!("weights directory does not exist: {}", dir.display()));
        }
    }

    if let Some(path) = &cfg.dataset {
        if !path.exists() {
            return Err(anyhow::anyhow!("dataset path does not exist: {}", path.display()));
        }
    }

    Ok(cfg)
}

fn resolve_user_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path;
    }

    if let Ok(caller_cwd) = std::env::var("FERRITE_RUN_CALLER_CWD") {
        let candidate = PathBuf::from(caller_cwd).join(&path);
        if candidate.exists() {
            return candidate;
        }
    }

    path
}

fn discover_files(dir: &PathBuf) -> Result<Vec<(PathBuf, usize)>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let meta = fs::metadata(&path)?;
        let len = meta.len() as usize;
        if len > 0 {
            files.push((path, len));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    if files.is_empty() {
        return Err(anyhow::anyhow!("no non-empty files found in weights dir"));
    }
    Ok(files)
}

fn chunk_file_sizes(files: &[(PathBuf, usize)], chunk_bytes: usize) -> Vec<usize> {
    let mut shards = Vec::new();
    for (_, mut len) in files {
        while len > 0 {
            let take = len.min(chunk_bytes);
            shards.push(take);
            len -= take;
        }
    }
    shards
}

fn detect_dataset_format(path: &PathBuf, hint: DatasetFormat) -> DatasetFormat {
    match hint {
        DatasetFormat::Auto => {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if ext == "jsonl" {
                DatasetFormat::Jsonl
            } else {
                DatasetFormat::Txt
            }
        }
        x => x,
    }
}

fn extract_json_string_value(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let key_pos = line.find(&needle)?;
    let mut i = key_pos + needle.len();
    let bytes = line.as_bytes();

    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1;

    let mut out = String::new();
    let mut escaped = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        i += 1;
        if escaped {
            match c {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                _ => out.push(c),
            }
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == '"' {
            return Some(out);
        }
        out.push(c);
    }
    None
}

fn load_dataset(path: &PathBuf, fmt: DatasetFormat, text_key: &str) -> Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match fmt {
            DatasetFormat::Txt => samples.push(trimmed.to_string()),
            DatasetFormat::Jsonl => {
                if let Some(text) = extract_json_string_value(trimmed, text_key) {
                    let t = text.trim();
                    if !t.is_empty() {
                        samples.push(t.to_string());
                    }
                }
            }
            DatasetFormat::Auto => {}
        }
    }

    if samples.is_empty() {
        return Err(anyhow::anyhow!(
            "dataset contains no usable samples (format={}, key={})",
            fmt.as_str(),
            text_key
        ));
    }
    Ok(samples)
}

fn encode_text_to_vec(text: &str, len: i64, seed: u64) -> Tensor {
    let mut out = vec![0f32; len as usize];
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return Tensor::from_slice(&out).view([len]);
    }

    let mut h = 1469598103934665603u64 ^ seed;
    for (i, &b) in bytes.iter().enumerate() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
        let idx = (h as usize + i) % out.len();
        let val = ((b as f32 / 255.0) * 2.0) - 1.0;
        out[idx] += val;
    }

    let n = (bytes.len() as f32).sqrt().max(1.0);
    for v in &mut out {
        *v /= n;
    }

    Tensor::from_slice(&out).view([len])
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    if !tch::Cuda::is_available() {
        println!("CUDA not available");
        return Ok(());
    }

    let runtime_cfg = PTXStableConfig {
        struct_size: std::mem::size_of::<PTXStableConfig>() as u32,
        abi_version: PTX_STABLE_ABI_VERSION,
        flags: 0,
        device_id: 0,
        pool_fraction: 0.70,
        fixed_pool_size: 0,
        reserve_vram: 256 * 1024 * 1024,
        max_streams: cfg.streams,
        quiet_init: 0,
        enable_leak_detection: 1,
        enable_pool_health: 1,
        _reserved0: 0,
    };
    let runtime = PtxRuntime::with_stable_config(0, Some(runtime_cfg))?;
    runtime.export_for_hook();
    runtime.export_context();

    init_pytorch_tlsf_ex(0, 0.70, cfg.streams).map_err(anyhow::Error::msg)?;

    let device = Device::Cuda(0);
    let active_streams = num_streams();
    let inv_sqrt_d = 1.0f64 / (cfg.hidden as f64).sqrt();
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f64;

    let mut directory_shard_bytes = Vec::new();
    let mut dataset_state: Option<DatasetState> = None;
    let mut dataset_format_used = DatasetFormat::Txt;

    let (shards_per_step, default_shard_bytes) = match cfg.weights_source {
        WeightsSource::Synthetic => {
            let model_bytes = cfg.model_gb * 1024 * 1024 * 1024;
            let shard_bytes = cfg.shard_mb * 1024 * 1024;
            (model_bytes.div_ceil(shard_bytes), shard_bytes)
        }
        WeightsSource::Directory => {
            let file_sizes = discover_files(cfg.weights_dir.as_ref().unwrap())?;
            let chunk_bytes = cfg.shard_mb * 1024 * 1024;
            directory_shard_bytes = chunk_file_sizes(&file_sizes, chunk_bytes);
            (directory_shard_bytes.len(), chunk_bytes)
        }
    };

    if let Some(path) = &cfg.dataset {
        dataset_format_used = detect_dataset_format(path, cfg.dataset_format);
        let samples = load_dataset(path, dataset_format_used, &cfg.text_key)?;
        dataset_state = Some(DatasetState::new(samples));
    }

    reset_peak(&runtime);
    let s0 = snap(&runtime);
    let pool_bytes = s0.total_pool;

    let largest_shard_bytes = match cfg.weights_source {
        WeightsSource::Synthetic => default_shard_bytes,
        WeightsSource::Directory => *directory_shard_bytes.iter().max().unwrap_or(&default_shard_bytes),
    };
    let state_multiplier = 2.8f64;
    let estimated_live =
        (largest_shard_bytes as f64 * state_multiplier * cfg.wave_streams as f64) as usize;
    let budget = (pool_bytes as f64 * 0.80) as usize;
    if estimated_live > budget {
        return Err(anyhow::anyhow!(
            "config too large: pool={:.1}MB estimated_live={:.1}MB (largest_shard={:.1}MB). Reduce --shard-mb or --wave-streams",
            pool_bytes as f64 / 1e6,
            estimated_live as f64 / 1e6,
            largest_shard_bytes as f64 / 1e6
        ));
    }

    println!("=== Ferrite TLSF: Streamed Fine-Tune Engine (Phase 1) ===\n");
    println!("  weights source:      {}", cfg.weights_source.as_str());
    if let Some(d) = &cfg.weights_dir {
        println!("  weights dir:         {}", d.display());
    }
    println!("  requested streams:   {}", cfg.streams);
    println!("  active streams:      {}", active_streams);
    println!("  wave streams:        {}", cfg.wave_streams);
    println!("  steps:               {}", cfg.steps);
    println!("  shards/step:         {}", shards_per_step);
    println!("  largest shard:       {:.1} MB", largest_shard_bytes as f64 / 1e6);
    println!("  micro-batch:         {}", cfg.micro_batch);
    println!("  hidden:              {}", cfg.hidden);
    println!("  lora rank:           {}", cfg.lora_rank);
    println!("  lora alpha:          {:.3}", cfg.lora_alpha);
    println!("  learning rate:       {:.6}", cfg.lr);
    println!(
        "  dataset:             {}",
        cfg.dataset
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none (synthetic)".to_string())
    );
    if let Some(ds) = &dataset_state {
        println!("  dataset format:      {}", dataset_format_used.as_str());
        println!("  dataset samples:     {}", ds.sample_count());
    }
    println!("  TLSF pool:           {:.1} MB\n", pool_bytes as f64 / 1e6);

    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let mut non_finite = 0usize;
    let mut step_times = Vec::with_capacity(cfg.steps);
    let t_total = Instant::now();

    for step in 0..cfg.steps {
        let t_step = Instant::now();
        let mut step_loss = 0.0f64;

        let mut shard_base = 0usize;
        while shard_base < shards_per_step {
            let wave = (shards_per_step - shard_base).min(cfg.wave_streams);

            for local_idx in 0..wave {
                let shard_idx = shard_base + local_idx;
                let stream_id = shard_idx % active_streams;
                set_torch_stream(stream_id);

                let shard_bytes = match cfg.weights_source {
                    WeightsSource::Synthetic => default_shard_bytes,
                    WeightsSource::Directory => directory_shard_bytes[shard_idx],
                };

                let elems = shard_bytes.div_ceil(std::mem::size_of::<f32>()) as i64;
                let rows = (elems / cfg.hidden).max(1);
                let used = rows * cfg.hidden;

                let base_w = (Tensor::randn([used], (Kind::Float, device)) * inv_sqrt_d)
                    .view([rows, cfg.hidden]);

                let mut lora_a = Tensor::randn([rows, cfg.lora_rank], (Kind::Float, device))
                    .set_requires_grad(true);
                let mut lora_b = Tensor::randn([cfg.lora_rank, cfg.hidden], (Kind::Float, device))
                    .set_requires_grad(true);

                let x = if let Some(ds) = &mut dataset_state {
                    let mut rows_x = Vec::with_capacity(cfg.micro_batch as usize);
                    for _ in 0..cfg.micro_batch {
                        let text = ds.next_text();
                        rows_x.push(encode_text_to_vec(text, cfg.hidden, 0xA11CE).view([1, cfg.hidden]));
                    }
                    Tensor::cat(&rows_x, 0).to_device(device)
                } else {
                    Tensor::randn([cfg.micro_batch, cfg.hidden], (Kind::Float, device))
                };

                let delta = lora_a.matmul(&lora_b) * lora_scale;
                let effective_w = &base_w + &delta;

                let y = (x.matmul(&effective_w.transpose(0, 1)) * inv_sqrt_d).tanh();
                let target = if let Some(ds) = &mut dataset_state {
                    let mut rows_t = Vec::with_capacity(cfg.micro_batch as usize);
                    for _ in 0..cfg.micro_batch {
                        let text = ds.next_text();
                        rows_t.push(encode_text_to_vec(text, rows, 0xBEEF).view([1, rows]));
                    }
                    Tensor::cat(&rows_t, 0).to_device(device).tanh()
                } else {
                    Tensor::randn([cfg.micro_batch, rows], (Kind::Float, device)).tanh()
                };

                let loss = (&y - &target).clamp(-10.0, 10.0).square().mean(Kind::Float);
                loss.backward();

                let ga = lora_a.grad().clamp(-1.0, 1.0);
                let gb = lora_b.grad().clamp(-1.0, 1.0);
                tch::no_grad(|| {
                    let next_a = &lora_a - ga * cfg.lr;
                    let next_b = &lora_b - gb * cfg.lr;
                    lora_a.copy_(&next_a);
                    lora_b.copy_(&next_b);
                });

                let lv = f64::try_from(&loss).unwrap_or(f64::NAN);
                if lv.is_finite() {
                    total_loss += lv;
                    step_loss += lv;
                    loss_count += 1;
                } else {
                    non_finite += 1;
                }
            }

            sync_all_streams();
            reset_torch_stream();
            shard_base += wave;
        }

        tch::Cuda::synchronize(0);
        let dt = t_step.elapsed().as_secs_f64();
        step_times.push(dt);

        if (step + 1) % 20 == 0 || step == 0 {
            let s = snap(&runtime);
            println!(
                "  step {:>4} | loss={:.6} | time={:.3}s | vram={:.0}MB peak={:.0}MB | frag={:.6}",
                step + 1,
                step_loss / shards_per_step as f64,
                dt,
                s.allocated as f64 / 1e6,
                s.peak as f64 / 1e6,
                s.frag
            );
        }
    }

    let wall = t_total.elapsed().as_secs_f64();
    let sf = snap(&runtime);
    let avg_step = step_times.iter().sum::<f64>() / step_times.len() as f64;
    let logical_streamed_gb = (shards_per_step * default_shard_bytes * cfg.steps) as f64
        / (1024.0 * 1024.0 * 1024.0);

    println!("\n  Fine-tune scaffold results:");
    println!("    Logical streamed:  {:.2} GB", logical_streamed_gb);
    println!("    Wall time:         {:.2}s", wall);
    println!("    Avg step:          {:.3}s", avg_step);
    println!(
        "    Avg loss:          {:.6}",
        if loss_count > 0 { total_loss / loss_count as f64 } else { f64::NAN }
    );
    println!("    Non-finite hits:   {}", non_finite);
    println!("    Peak VRAM:         {:.1} MB", sf.peak as f64 / 1e6);
    println!("    TLSF allocs:       {}", sf.allocs - s0.allocs);
    println!("    TLSF frees:        {}", sf.frees - s0.frees);
    println!("    Fragmentation:     {:.6}", sf.frag);
    println!("    Pool healthy:      {}", if runtime.validate_pool().is_valid { "YES" } else { "NO" });

    println!("\n  Numeric results:");
    println!("RESULT mode=streamed_finetune_phase1");
    println!("RESULT weights_source={}", cfg.weights_source.as_str());
    println!(
        "RESULT dataset={}",
        cfg.dataset
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "RESULT dataset_format={}",
        if dataset_state.is_some() {
            dataset_format_used.as_str()
        } else {
            "none"
        }
    );
    println!(
        "RESULT dataset_samples={}",
        dataset_state.as_ref().map(|d| d.sample_count()).unwrap_or(0)
    );
    println!("RESULT streams={}", active_streams);
    println!("RESULT wave_streams={}", cfg.wave_streams);
    println!("RESULT steps={}", cfg.steps);
    println!("RESULT shards_per_step={}", shards_per_step);
    println!("RESULT micro_batch={}", cfg.micro_batch);
    println!("RESULT hidden={}", cfg.hidden);
    println!("RESULT lora_rank={}", cfg.lora_rank);
    println!("RESULT lora_alpha={:.6}", cfg.lora_alpha);
    println!("RESULT lr={:.9}", cfg.lr);
    println!("RESULT pool_mb={:.6}", pool_bytes as f64 / 1e6);
    println!("RESULT streamed_gb_total={:.6}", logical_streamed_gb);
    println!("RESULT wall_s={:.6}", wall);
    println!("RESULT avg_step_s={:.6}", avg_step);
    println!(
        "RESULT avg_loss={:.9}",
        if loss_count > 0 { total_loss / loss_count as f64 } else { f64::NAN }
    );
    println!("RESULT non_finite_events={}", non_finite);
    println!("RESULT peak_vram_mb={:.6}", sf.peak as f64 / 1e6);
    println!("RESULT tlsf_allocs={}", sf.allocs - s0.allocs);
    println!("RESULT tlsf_frees={}", sf.frees - s0.frees);
    println!("RESULT fragmentation={:.9}", sf.frag);
    println!("RESULT pool_healthy={}", if runtime.validate_pool().is_valid { 1 } else { 0 });

    Ok(())
}
