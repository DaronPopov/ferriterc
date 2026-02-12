use std::fs;
use std::f64::consts::PI;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

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
const DEFAULT_OPTIMIZER: &str = "adamw";
const DEFAULT_SCHEDULE: &str = "cosine_decay";
const DEFAULT_WEIGHT_DECAY: f64 = 0.01;
const DEFAULT_BETA1: f64 = 0.9;
const DEFAULT_BETA2: f64 = 0.999;
const DEFAULT_EPS: f64 = 1e-8;
const DEFAULT_MOMENTUM: f64 = 0.9;
const DEFAULT_INNER_STEPS: usize = 1;
const DEFAULT_GRAD_CLIP: f64 = 1.0;
const DEFAULT_MIN_LR_FACTOR: f64 = 0.10;
const MAGIC: &[u8; 8] = b"FRTA_CKP";
const VERSION: u32 = 2;
const MIN_SUPPORTED_VERSION: u32 = 1;
const CHECKPOINT_HEADER_BYTES: usize = 80;

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

#[derive(Clone, Copy)]
enum OptimizerKind {
    Sgd,
    AdamW,
}

impl OptimizerKind {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "sgd" => Ok(Self::Sgd),
            "adamw" => Ok(Self::AdamW),
            _ => Err(anyhow::anyhow!("invalid --optimizer: {s} (expected sgd|adamw)")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Sgd => "sgd",
            Self::AdamW => "adamw",
        }
    }
}

#[derive(Clone, Copy)]
enum ScheduleKind {
    Constant,
    LinearWarmup,
    CosineDecay,
    OneCycle,
}

impl ScheduleKind {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "constant" => Ok(Self::Constant),
            "linear_warmup" => Ok(Self::LinearWarmup),
            "cosine_decay" => Ok(Self::CosineDecay),
            "one_cycle" => Ok(Self::OneCycle),
            _ => Err(anyhow::anyhow!(
                "invalid --schedule: {s} (expected constant|linear_warmup|cosine_decay|one_cycle)"
            )),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::LinearWarmup => "linear_warmup",
            Self::CosineDecay => "cosine_decay",
            Self::OneCycle => "one_cycle",
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
    min_lr: f64,
    warmup_steps: usize,
    schedule: ScheduleKind,
    optimizer: OptimizerKind,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    momentum: f64,
    grad_clip: f64,
    inner_steps: usize,
    checkpoint_path: Option<PathBuf>,
    save_every: usize,
    resume: bool,
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
            min_lr: DEFAULT_LR * DEFAULT_MIN_LR_FACTOR,
            warmup_steps: 0,
            schedule: ScheduleKind::parse(DEFAULT_SCHEDULE).expect("valid default schedule"),
            optimizer: OptimizerKind::parse(DEFAULT_OPTIMIZER).expect("valid default optimizer"),
            weight_decay: DEFAULT_WEIGHT_DECAY,
            beta1: DEFAULT_BETA1,
            beta2: DEFAULT_BETA2,
            eps: DEFAULT_EPS,
            momentum: DEFAULT_MOMENTUM,
            grad_clip: DEFAULT_GRAD_CLIP,
            inner_steps: DEFAULT_INNER_STEPS,
            checkpoint_path: None,
            save_every: 0,
            resume: false,
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
    let _ = runtime.sync_all();
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
            "--min-lr" => cfg.min_lr = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --min-lr"))?.parse()?,
            "--warmup-steps" => cfg.warmup_steps = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --warmup-steps"))?.parse()?,
            "--schedule" => {
                cfg.schedule = ScheduleKind::parse(
                    &args.next().ok_or_else(|| anyhow::anyhow!("missing value for --schedule"))?,
                )?;
            }
            "--optimizer" => {
                cfg.optimizer = OptimizerKind::parse(
                    &args.next().ok_or_else(|| anyhow::anyhow!("missing value for --optimizer"))?,
                )?;
            }
            "--weight-decay" => cfg.weight_decay = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --weight-decay"))?.parse()?,
            "--beta1" => cfg.beta1 = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --beta1"))?.parse()?,
            "--beta2" => cfg.beta2 = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --beta2"))?.parse()?,
            "--eps" => cfg.eps = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --eps"))?.parse()?,
            "--momentum" => cfg.momentum = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --momentum"))?.parse()?,
            "--grad-clip" => cfg.grad_clip = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --grad-clip"))?.parse()?,
            "--inner-steps" => cfg.inner_steps = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --inner-steps"))?.parse()?,
            "--checkpoint-path" => {
                cfg.checkpoint_path = Some(PathBuf::from(
                    args.next().ok_or_else(|| anyhow::anyhow!("missing value for --checkpoint-path"))?,
                ));
            }
            "--save-every" => cfg.save_every = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --save-every"))?.parse()?,
            "--resume" => cfg.resume = true,
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
                println!("  --lora-rank N --lora-alpha F --lr F --min-lr F --warmup-steps N");
                println!("  --optimizer sgd|adamw --schedule constant|linear_warmup|cosine_decay|one_cycle");
                println!("  --weight-decay F --beta1 F --beta2 F --eps F --momentum F");
                println!("  --grad-clip F --inner-steps N");
                println!("  --checkpoint-path PATH --save-every N --resume");
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
        || cfg.min_lr <= 0.0
        || cfg.inner_steps == 0
        || cfg.grad_clip <= 0.0
        || cfg.weight_decay < 0.0
        || cfg.beta1 <= 0.0
        || cfg.beta1 >= 1.0
        || cfg.beta2 <= 0.0
        || cfg.beta2 >= 1.0
        || cfg.eps <= 0.0
        || cfg.momentum < 0.0
        || cfg.momentum >= 1.0
    {
        return Err(anyhow::anyhow!("invalid numeric hyperparameters"));
    }

    if cfg.min_lr > cfg.lr {
        return Err(anyhow::anyhow!("--min-lr must be <= --lr"));
    }

    if let Some(path) = cfg.weights_dir.clone() {
        cfg.weights_dir = Some(resolve_user_path(path));
    }
    if let Some(path) = cfg.checkpoint_path.clone() {
        cfg.checkpoint_path = Some(resolve_user_path(path));
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

    if cfg.resume && cfg.checkpoint_path.is_none() {
        return Err(anyhow::anyhow!("--resume requires --checkpoint-path"));
    }
    if cfg.resume {
        let cp = cfg.checkpoint_path.as_ref().expect("checked above");
        if !cp.exists() {
            return Err(anyhow::anyhow!(
                "--resume requested but checkpoint does not exist: {}",
                cp.display()
            ));
        }
    }
    if cfg.save_every > 0 && cfg.checkpoint_path.is_none() {
        return Err(anyhow::anyhow!(
            "--save-every requires --checkpoint-path"
        ));
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

fn synthetic_base_weight(rows: i64, hidden: i64, shard_idx: usize, device: Device, inv_sqrt_d: f64) -> Tensor {
    let used = rows * hidden;
    let idx = Tensor::arange(used, (Kind::Float, device));
    let phase = (shard_idx as f64 + 1.0) * 0.0137;
    let wave = (&idx * 0.011 + phase).sin() + (&idx * 0.017 + phase * 1.7).cos();
    (wave * (0.5 * inv_sqrt_d)).view([rows, hidden])
}

fn resolved_warmup_steps(cfg: &Config, total_updates: usize) -> usize {
    if total_updates <= 1 {
        return 0;
    }
    if cfg.warmup_steps == 0 {
        (total_updates / 10).max(1)
    } else {
        cfg.warmup_steps.min(total_updates - 1)
    }
}

fn lr_at_update(cfg: &Config, update_idx: usize, total_updates: usize, warmup_steps: usize) -> f64 {
    match cfg.schedule {
        ScheduleKind::Constant => cfg.lr,
        ScheduleKind::LinearWarmup => {
            if warmup_steps == 0 {
                cfg.lr
            } else if update_idx >= warmup_steps {
                cfg.lr
            } else {
                cfg.lr * (update_idx as f64 / warmup_steps as f64)
            }
        }
        ScheduleKind::CosineDecay => {
            if warmup_steps > 0 && update_idx < warmup_steps {
                return cfg.lr * (update_idx as f64 / warmup_steps as f64);
            }
            let decay_total = total_updates.saturating_sub(warmup_steps).max(1);
            let progress = (update_idx.saturating_sub(warmup_steps) as f64 / decay_total as f64).min(1.0);
            let cosine = 0.5 * (1.0 + (PI * progress).cos());
            cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine
        }
        ScheduleKind::OneCycle => {
            let up_steps = ((total_updates as f64) * 0.30) as usize;
            let up_steps = up_steps.max(1).min(total_updates);
            let down_steps = total_updates.saturating_sub(up_steps).max(1);
            if update_idx < up_steps {
                let t = update_idx as f64 / up_steps as f64;
                let cosine = 0.5 * (1.0 + (PI * (1.0 + t)).cos());
                cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine
            } else {
                let t = (update_idx - up_steps) as f64 / down_steps as f64;
                let cosine = 0.5 * (1.0 + (PI * t).cos());
                cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine
            }
        }
    }
}

fn clip_pair_by_global_norm(ga: Tensor, gb: Tensor, max_norm: f64) -> (Tensor, Tensor, f64, bool) {
    let ga_sq = (&ga * &ga).sum(Kind::Float);
    let gb_sq = (&gb * &gb).sum(Kind::Float);
    let grad_norm = (&ga_sq + &gb_sq).sqrt().double_value(&[]);
    if !grad_norm.is_finite() || grad_norm <= max_norm {
        return (ga, gb, grad_norm, false);
    }
    let scale = max_norm / (grad_norm + 1e-6);
    (ga * scale, gb * scale, grad_norm, true)
}

struct ShardOptimState {
    step: usize,
    m_a: Tensor,
    m_b: Tensor,
    v_a: Tensor,
    v_b: Tensor,
    vel_a: Tensor,
    vel_b: Tensor,
}

impl ShardOptimState {
    fn new(lora_a: &Tensor, lora_b: &Tensor) -> Self {
        Self {
            step: 0,
            m_a: Tensor::zeros_like(lora_a),
            m_b: Tensor::zeros_like(lora_b),
            v_a: Tensor::zeros_like(lora_a),
            v_b: Tensor::zeros_like(lora_b),
            vel_a: Tensor::zeros_like(lora_a),
            vel_b: Tensor::zeros_like(lora_b),
        }
    }
}

fn apply_optimizer_step(
    cfg: &Config,
    state: &mut ShardOptimState,
    lora_a: &mut Tensor,
    lora_b: &mut Tensor,
    ga: &Tensor,
    gb: &Tensor,
    lr: f64,
) {
    tch::no_grad(|| {
        match cfg.optimizer {
            OptimizerKind::Sgd => {
                state.vel_a = &state.vel_a * cfg.momentum + ga;
                state.vel_b = &state.vel_b * cfg.momentum + gb;

                let update_a = &state.vel_a + lora_a.shallow_clone() * cfg.weight_decay;
                let update_b = &state.vel_b + lora_b.shallow_clone() * cfg.weight_decay;

                let next_a = lora_a.shallow_clone() - update_a * lr;
                let next_b = lora_b.shallow_clone() - update_b * lr;
                lora_a.copy_(&next_a);
                lora_b.copy_(&next_b);
            }
            OptimizerKind::AdamW => {
                state.step = state.step.saturating_add(1);
                state.m_a = &state.m_a * cfg.beta1 + ga * (1.0 - cfg.beta1);
                state.m_b = &state.m_b * cfg.beta1 + gb * (1.0 - cfg.beta1);
                state.v_a = &state.v_a * cfg.beta2 + (ga * ga) * (1.0 - cfg.beta2);
                state.v_b = &state.v_b * cfg.beta2 + (gb * gb) * (1.0 - cfg.beta2);

                let bias_c1 = 1.0 - cfg.beta1.powf(state.step as f64);
                let bias_c2 = 1.0 - cfg.beta2.powf(state.step as f64);

                let step_a = (&state.m_a / bias_c1) / ((&state.v_a / bias_c2).sqrt() + cfg.eps);
                let step_b = (&state.m_b / bias_c1) / ((&state.v_b / bias_c2).sqrt() + cfg.eps);

                let decayed_a = lora_a.shallow_clone() * (1.0 - lr * cfg.weight_decay);
                let decayed_b = lora_b.shallow_clone() * (1.0 - lr * cfg.weight_decay);
                let next_a = decayed_a - step_a * lr;
                let next_b = decayed_b - step_b * lr;
                lora_a.copy_(&next_a);
                lora_b.copy_(&next_b);
            }
        }
    });

    if matches!(cfg.optimizer, OptimizerKind::Sgd) {
        state.step = state.step.saturating_add(1);
    }
}

struct ShardTrainState {
    rows: i64,
    lora_a_cpu: Tensor,
    lora_b_cpu: Tensor,
    optim_cpu: ShardOptimState,
}

impl ShardTrainState {
    fn new(rows: i64, cfg: &Config) -> Self {
        let cpu = Device::Cpu;
        let lora_a_cpu = (Tensor::randn([rows, cfg.lora_rank], (Kind::Float, cpu)) * 0.02).set_requires_grad(true);
        let lora_b_cpu = (Tensor::zeros([cfg.lora_rank, cfg.hidden], (Kind::Float, cpu))).set_requires_grad(true);
        let optim_cpu = ShardOptimState::new(&lora_a_cpu, &lora_b_cpu);
        Self {
            rows,
            lora_a_cpu,
            lora_b_cpu,
            optim_cpu,
        }
    }

    fn to_device(&self, device: Device) -> (Tensor, Tensor, ShardOptimState) {
        let lora_a = self
            .lora_a_cpu
            .to_device(device)
            .detach()
            .set_requires_grad(true);
        let lora_b = self
            .lora_b_cpu
            .to_device(device)
            .detach()
            .set_requires_grad(true);
        let optim_state = ShardOptimState {
            step: self.optim_cpu.step,
            m_a: self.optim_cpu.m_a.to_device(device),
            m_b: self.optim_cpu.m_b.to_device(device),
            v_a: self.optim_cpu.v_a.to_device(device),
            v_b: self.optim_cpu.v_b.to_device(device),
            vel_a: self.optim_cpu.vel_a.to_device(device),
            vel_b: self.optim_cpu.vel_b.to_device(device),
        };
        (lora_a, lora_b, optim_state)
    }

    fn update_from_device(&mut self, lora_a: Tensor, lora_b: Tensor, optim_state: ShardOptimState) {
        let cpu = Device::Cpu;
        self.lora_a_cpu = lora_a.detach().to_device(cpu).set_requires_grad(true);
        self.lora_b_cpu = lora_b.detach().to_device(cpu).set_requires_grad(true);
        self.optim_cpu = ShardOptimState {
            step: optim_state.step,
            m_a: optim_state.m_a.to_device(cpu),
            m_b: optim_state.m_b.to_device(cpu),
            v_a: optim_state.v_a.to_device(cpu),
            v_b: optim_state.v_b.to_device(cpu),
            vel_a: optim_state.vel_a.to_device(cpu),
            vel_b: optim_state.vel_b.to_device(cpu),
        };
    }
}

#[derive(Debug)]
struct AdapterCheckpoint {
    version: u32,
    step: usize,
    loss: f64,
    lora_rank: i64,
    lora_alpha: f64,
    lr: f64,
    hidden: i64,
    shard_count: usize,
    adapters_a: Vec<Tensor>,
    adapters_b: Vec<Tensor>,
    optim_m_a: Vec<Tensor>,
    optim_m_b: Vec<Tensor>,
    optim_v_a: Vec<Tensor>,
    optim_v_b: Vec<Tensor>,
    optim_vel_a: Vec<Tensor>,
    optim_vel_b: Vec<Tensor>,
}

#[derive(Clone, Debug)]
struct CheckpointHeader {
    version: u32,
    step: u64,
    loss_bits: u64,
    lora_rank: u32,
    hidden: u32,
    shard_count: u32,
    lora_alpha_bits: u64,
    lr_bits: u64,
    timestamp: u64,
    tensor_count: u32,
}

impl CheckpointHeader {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(CHECKPOINT_HEADER_BYTES);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.step.to_le_bytes());
        buf.extend_from_slice(&self.loss_bits.to_le_bytes());
        buf.extend_from_slice(&self.lora_rank.to_le_bytes());
        buf.extend_from_slice(&self.hidden.to_le_bytes());
        buf.extend_from_slice(&self.shard_count.to_le_bytes());
        buf.extend_from_slice(&self.lora_alpha_bits.to_le_bytes());
        buf.extend_from_slice(&self.lr_bits.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.tensor_count.to_le_bytes());
        buf.resize(CHECKPOINT_HEADER_BYTES, 0);
        buf
    }

    fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < CHECKPOINT_HEADER_BYTES {
            return Err(anyhow::anyhow!("header too short: {} bytes", data.len()));
        }
        if &data[0..8] != MAGIC {
            return Err(anyhow::anyhow!("invalid checkpoint magic"));
        }
        Ok(Self {
            version: u32::from_le_bytes(data[8..12].try_into()?),
            step: u64::from_le_bytes(data[12..20].try_into()?),
            loss_bits: u64::from_le_bytes(data[20..28].try_into()?),
            lora_rank: u32::from_le_bytes(data[28..32].try_into()?),
            hidden: u32::from_le_bytes(data[32..36].try_into()?),
            shard_count: u32::from_le_bytes(data[36..40].try_into()?),
            lora_alpha_bits: u64::from_le_bytes(data[40..48].try_into()?),
            lr_bits: u64::from_le_bytes(data[48..56].try_into()?),
            timestamp: u64::from_le_bytes(data[56..64].try_into()?),
            tensor_count: u32::from_le_bytes(data[64..68].try_into()?),
        })
    }
}

#[derive(Clone, Debug)]
struct TensorMeta {
    name_len: u16,
    name: String,
    ndim: u8,
    dims: Vec<i64>,
    dtype: u8,
    data_bytes: u64,
}

impl TensorMeta {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.name_len.to_le_bytes());
        buf.extend_from_slice(self.name.as_bytes());
        buf.push(self.ndim);
        for &d in &self.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.push(self.dtype);
        buf.extend_from_slice(&self.data_bytes.to_le_bytes());
        buf
    }

    fn from_reader(data: &[u8], offset: &mut usize) -> Result<Self> {
        let name_len = u16::from_le_bytes(read_chunk(data, offset, 2)?.try_into()?);
        let name = String::from_utf8(read_chunk(data, offset, name_len as usize)?.to_vec())?;
        let ndim = read_chunk(data, offset, 1)?[0];
        let mut dims = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            dims.push(i64::from_le_bytes(read_chunk(data, offset, 8)?.try_into()?));
        }
        let dtype = read_chunk(data, offset, 1)?[0];
        let data_bytes = u64::from_le_bytes(read_chunk(data, offset, 8)?.try_into()?);
        Ok(Self {
            name_len,
            name,
            ndim,
            dims,
            dtype,
            data_bytes,
        })
    }
}

fn read_chunk<'a>(data: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = (*offset).saturating_add(len);
    if end > data.len() {
        return Err(anyhow::anyhow!(
            "checkpoint truncated: need {} bytes at offset {}, total {}",
            len,
            *offset,
            data.len()
        ));
    }
    let chunk = &data[*offset..end];
    *offset = end;
    Ok(chunk)
}

fn decode_f32_bytes(raw: &[u8]) -> Result<Vec<f32>> {
    if !raw.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(anyhow::anyhow!(
            "invalid f32 blob size: {} bytes",
            raw.len()
        ));
    }
    let mut out = Vec::with_capacity(raw.len() / std::mem::size_of::<f32>());
    for bytes in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    }
    Ok(out)
}

fn write_named_tensor(writer: &mut BufWriter<fs::File>, name: String, tensor: &Tensor) -> Result<()> {
    let t = tensor.to_device(Device::Cpu).to_kind(Kind::Float);
    let size = t.size();
    let numel: i64 = size.iter().product();
    let data_bytes = (numel as usize) * std::mem::size_of::<f32>();
    let meta = TensorMeta {
        name_len: name.len() as u16,
        name,
        ndim: size.len() as u8,
        dims: size,
        dtype: 0,
        data_bytes: data_bytes as u64,
    };
    writer.write_all(&meta.to_bytes())?;
    let flat = Vec::<f32>::try_from(t.flatten(0, -1))?;
    for v in flat {
        writer.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn save_checkpoint(ckpt: &AdapterCheckpoint, path: &PathBuf) -> Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let tensor_count = (ckpt.adapters_a.len()
        + ckpt.adapters_b.len()
        + ckpt.optim_m_a.len()
        + ckpt.optim_m_b.len()
        + ckpt.optim_v_a.len()
        + ckpt.optim_v_b.len()
        + ckpt.optim_vel_a.len()
        + ckpt.optim_vel_b.len()) as u32;
    let header = CheckpointHeader {
        version: ckpt.version,
        step: ckpt.step as u64,
        loss_bits: ckpt.loss.to_bits(),
        lora_rank: ckpt.lora_rank as u32,
        hidden: ckpt.hidden as u32,
        shard_count: ckpt.shard_count as u32,
        lora_alpha_bits: ckpt.lora_alpha.to_bits(),
        lr_bits: ckpt.lr.to_bits(),
        timestamp,
        tensor_count,
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&header.to_bytes())?;

    for (i, tensor) in ckpt.adapters_a.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_lora_a"), tensor)?;
    }

    for (i, tensor) in ckpt.adapters_b.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_lora_b"), tensor)?;
    }

    for (i, tensor) in ckpt.optim_m_a.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_optim_m_a"), tensor)?;
    }
    for (i, tensor) in ckpt.optim_m_b.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_optim_m_b"), tensor)?;
    }
    for (i, tensor) in ckpt.optim_v_a.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_optim_v_a"), tensor)?;
    }
    for (i, tensor) in ckpt.optim_v_b.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_optim_v_b"), tensor)?;
    }
    for (i, tensor) in ckpt.optim_vel_a.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_optim_vel_a"), tensor)?;
    }
    for (i, tensor) in ckpt.optim_vel_b.iter().enumerate() {
        write_named_tensor(&mut writer, format!("shard_{i}_optim_vel_b"), tensor)?;
    }

    writer.flush()?;
    Ok(())
}

fn load_checkpoint(path: &PathBuf, device: Device) -> Result<AdapterCheckpoint> {
    let data = fs::read(path)?;
    let header = CheckpointHeader::from_bytes(&data)?;
    if header.version < MIN_SUPPORTED_VERSION || header.version > VERSION {
        return Err(anyhow::anyhow!(
            "checkpoint version mismatch: file={} supported={}..={}",
            header.version,
            MIN_SUPPORTED_VERSION,
            VERSION
        ));
    }

    let mut offset = CHECKPOINT_HEADER_BYTES;
    let mut adapters_a = Vec::new();
    let mut adapters_b = Vec::new();
    let mut optim_m_a = Vec::new();
    let mut optim_m_b = Vec::new();
    let mut optim_v_a = Vec::new();
    let mut optim_v_b = Vec::new();
    let mut optim_vel_a = Vec::new();
    let mut optim_vel_b = Vec::new();

    for _ in 0..header.tensor_count {
        let meta = TensorMeta::from_reader(&data, &mut offset)?;
        let byte_count = meta.data_bytes as usize;
        let raw = read_chunk(&data, &mut offset, byte_count)?;
        let floats = decode_f32_bytes(raw)?;

        let requires_grad = meta.name.ends_with("_lora_a") || meta.name.ends_with("_lora_b");
        let tensor = Tensor::from_slice(&floats)
            .reshape(&meta.dims)
            .to_device(device)
            .set_requires_grad(requires_grad);
        if meta.name.ends_with("_lora_a") {
            adapters_a.push(tensor);
        } else if meta.name.ends_with("_lora_b") {
            adapters_b.push(tensor);
        } else if meta.name.ends_with("_optim_m_a") {
            optim_m_a.push(tensor);
        } else if meta.name.ends_with("_optim_m_b") {
            optim_m_b.push(tensor);
        } else if meta.name.ends_with("_optim_v_a") {
            optim_v_a.push(tensor);
        } else if meta.name.ends_with("_optim_v_b") {
            optim_v_b.push(tensor);
        } else if meta.name.ends_with("_optim_vel_a") {
            optim_vel_a.push(tensor);
        } else if meta.name.ends_with("_optim_vel_b") {
            optim_vel_b.push(tensor);
        }
    }

    Ok(AdapterCheckpoint {
        version: header.version,
        step: header.step as usize,
        loss: f64::from_bits(header.loss_bits),
        lora_rank: header.lora_rank as i64,
        lora_alpha: f64::from_bits(header.lora_alpha_bits),
        lr: f64::from_bits(header.lr_bits),
        hidden: header.hidden as i64,
        shard_count: header.shard_count as usize,
        adapters_a,
        adapters_b,
        optim_m_a,
        optim_m_b,
        optim_v_a,
        optim_v_b,
        optim_vel_a,
        optim_vel_b,
    })
}

fn save_training_state(
    cfg: &Config,
    path: &PathBuf,
    global_step: usize,
    avg_loss: f64,
    shard_states: &[ShardTrainState],
) -> Result<()> {
    let adapters_a: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.lora_a_cpu.shallow_clone())
        .collect();
    let adapters_b: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.lora_b_cpu.shallow_clone())
        .collect();
    let optim_m_a: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.optim_cpu.m_a.shallow_clone())
        .collect();
    let optim_m_b: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.optim_cpu.m_b.shallow_clone())
        .collect();
    let optim_v_a: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.optim_cpu.v_a.shallow_clone())
        .collect();
    let optim_v_b: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.optim_cpu.v_b.shallow_clone())
        .collect();
    let optim_vel_a: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.optim_cpu.vel_a.shallow_clone())
        .collect();
    let optim_vel_b: Vec<Tensor> = shard_states
        .iter()
        .map(|s| s.optim_cpu.vel_b.shallow_clone())
        .collect();
    let ckpt = AdapterCheckpoint {
        version: VERSION,
        step: global_step,
        loss: avg_loss,
        lora_rank: cfg.lora_rank,
        lora_alpha: cfg.lora_alpha,
        lr: cfg.lr,
        hidden: cfg.hidden,
        shard_count: shard_states.len(),
        adapters_a,
        adapters_b,
        optim_m_a,
        optim_m_b,
        optim_v_a,
        optim_v_b,
        optim_vel_a,
        optim_vel_b,
    };
    save_checkpoint(&ckpt, path)
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    aten_ptx::ensure_libtorch_cuda_loaded();
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
    if active_streams == 0 {
        return Err(anyhow::anyhow!("runtime reported zero active CUDA streams"));
    }
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

    let shard_bytes_vec: Vec<usize> = match cfg.weights_source {
        WeightsSource::Synthetic => vec![default_shard_bytes; shards_per_step],
        WeightsSource::Directory => directory_shard_bytes.clone(),
    };
    let shard_rows_vec: Vec<i64> = shard_bytes_vec
        .iter()
        .map(|&bytes| {
            let elems = bytes.div_ceil(std::mem::size_of::<f32>()) as i64;
            (elems / cfg.hidden).max(1)
        })
        .collect();
    let mut shard_states: Vec<ShardTrainState> = shard_rows_vec
        .iter()
        .map(|&rows| ShardTrainState::new(rows, &cfg))
        .collect();

    let mut start_step = 0usize;
    let mut resumed = false;
    let mut resumed_checkpoint_version = 0u32;
    let mut restored_optim_states = 0usize;
    if cfg.resume {
        let cp = cfg
            .checkpoint_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--resume requires --checkpoint-path"))?;
        let loaded = load_checkpoint(cp, Device::Cpu)?;
        if loaded.hidden != cfg.hidden || loaded.lora_rank != cfg.lora_rank {
            return Err(anyhow::anyhow!(
                "checkpoint shape mismatch: hidden/rank in checkpoint ({}/{}) != config ({}/{})",
                loaded.hidden,
                loaded.lora_rank,
                cfg.hidden,
                cfg.lora_rank
            ));
        }
        let restore_count = shard_states
            .len()
            .min(loaded.adapters_a.len())
            .min(loaded.adapters_b.len());
        let mut restored = 0usize;
        for i in 0..restore_count {
            let expected_a = [shard_states[i].rows, cfg.lora_rank];
            let expected_b = [cfg.lora_rank, cfg.hidden];
            if loaded.adapters_a[i].size() == expected_a && loaded.adapters_b[i].size() == expected_b {
                shard_states[i].lora_a_cpu = loaded.adapters_a[i].shallow_clone().to_device(Device::Cpu).set_requires_grad(true);
                shard_states[i].lora_b_cpu = loaded.adapters_b[i].shallow_clone().to_device(Device::Cpu).set_requires_grad(true);
                let mut optim_state = ShardOptimState::new(&shard_states[i].lora_a_cpu, &shard_states[i].lora_b_cpu);
                let expected_ma = expected_a;
                let expected_mb = expected_b;
                let expected_va = expected_a;
                let expected_vb = expected_b;
                let expected_vela = expected_a;
                let expected_velb = expected_b;
                let has_optim_state = loaded.optim_m_a.get(i).is_some()
                    && loaded.optim_m_b.get(i).is_some()
                    && loaded.optim_v_a.get(i).is_some()
                    && loaded.optim_v_b.get(i).is_some()
                    && loaded.optim_vel_a.get(i).is_some()
                    && loaded.optim_vel_b.get(i).is_some()
                    && loaded.optim_m_a[i].size() == expected_ma
                    && loaded.optim_m_b[i].size() == expected_mb
                    && loaded.optim_v_a[i].size() == expected_va
                    && loaded.optim_v_b[i].size() == expected_vb
                    && loaded.optim_vel_a[i].size() == expected_vela
                    && loaded.optim_vel_b[i].size() == expected_velb;
                if has_optim_state {
                    optim_state.m_a = loaded.optim_m_a[i].to_device(Device::Cpu);
                    optim_state.m_b = loaded.optim_m_b[i].to_device(Device::Cpu);
                    optim_state.v_a = loaded.optim_v_a[i].to_device(Device::Cpu);
                    optim_state.v_b = loaded.optim_v_b[i].to_device(Device::Cpu);
                    optim_state.vel_a = loaded.optim_vel_a[i].to_device(Device::Cpu);
                    optim_state.vel_b = loaded.optim_vel_b[i].to_device(Device::Cpu);
                    restored_optim_states += 1;
                }
                optim_state.step = loaded.step.saturating_mul(cfg.inner_steps);
                shard_states[i].optim_cpu = optim_state;
                restored += 1;
            }
        }
        start_step = loaded.step.min(cfg.steps);
        resumed = true;
        resumed_checkpoint_version = loaded.version;
        println!(
            "[checkpoint] resumed from {} (version={}, step={}, restored_shards={}/{}, optimizer_state={}/{})",
            cp.display(),
            loaded.version,
            loaded.step,
            restored,
            shard_states.len(),
            restored_optim_states,
            shard_states.len()
        );
    }

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

    let checkpoint_version_runtime = if resumed {
        resumed_checkpoint_version
    } else {
        VERSION
    };

    println!("=== Ferrite TLSF: Streamed Fine-Tune Engine (Phase 3) ===\n");
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
    println!("  learning rate:       {:.6} (min {:.6})", cfg.lr, cfg.min_lr);
    println!("  optimizer:           {}", cfg.optimizer.as_str());
    println!("  schedule:            {}", cfg.schedule.as_str());
    println!("  weight decay:        {:.6}", cfg.weight_decay);
    println!("  inner steps:         {}", cfg.inner_steps);
    println!("  grad clip norm:      {:.3}", cfg.grad_clip);
    println!("  checkpoint path:     {}", cfg.checkpoint_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".to_string()));
    println!("  checkpoint version:  {}", checkpoint_version_runtime);
    println!("  save every:          {}", cfg.save_every);
    println!("  resume:              {}", if resumed { "yes" } else { "no" });
    println!("  start step:          {}", start_step);
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
    let total_updates = cfg
        .steps
        .saturating_mul(shards_per_step)
        .saturating_mul(cfg.inner_steps)
        .max(1);
    let initial_update_cursor = start_step
        .saturating_mul(shards_per_step)
        .saturating_mul(cfg.inner_steps);
    let warmup_steps = resolved_warmup_steps(&cfg, total_updates);
    println!("  total updates:       {}", total_updates);
    println!("  warmup updates:      {}", warmup_steps);
    println!("  update cursor start: {}", initial_update_cursor);
    println!("  TLSF pool:           {:.1} MB\n", pool_bytes as f64 / 1e6);

    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let mut non_finite = 0usize;
    let mut update_cursor = initial_update_cursor;
    let mut updates_applied = 0usize;
    let mut executed_steps = 0usize;
    let mut clip_events = 0usize;
    let mut grad_norm_sum = 0.0f64;
    let mut grad_norm_count = 0usize;
    let mut lr_sum = 0.0f64;
    let mut lr_min_seen = f64::MAX;
    let mut lr_max_seen = f64::MIN;
    let mut step_times = Vec::with_capacity(cfg.steps.saturating_sub(start_step));
    let t_total = Instant::now();

    for step in start_step..cfg.steps {
        executed_steps += 1;
        let t_step = Instant::now();
        let mut step_loss = 0.0f64;
        let mut step_updates = 0usize;
        let mut step_lr_sum = 0.0f64;
        let mut step_clip_events = 0usize;

        let mut shard_base = 0usize;
        while shard_base < shards_per_step {
            let wave = (shards_per_step - shard_base).min(cfg.wave_streams);

            for local_idx in 0..wave {
                let shard_idx = shard_base + local_idx;
                let stream_id = shard_idx % active_streams;
                set_torch_stream(stream_id);

                let rows = shard_states[shard_idx].rows;
                let base_w = synthetic_base_weight(rows, cfg.hidden, shard_idx, device, inv_sqrt_d);

                let (mut lora_a, mut lora_b, mut optim_state) = shard_states[shard_idx].to_device(device);

                for _ in 0..cfg.inner_steps {
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
                    let lv = f64::try_from(&loss).unwrap_or(f64::NAN);
                    if !lv.is_finite() {
                        non_finite += 1;
                        continue;
                    }

                    loss.backward();

                    let current_lr = lr_at_update(&cfg, update_cursor, total_updates, warmup_steps);
                    let ga_raw = lora_a.grad();
                    let gb_raw = lora_b.grad();
                    let (ga, gb, grad_norm, was_clipped) =
                        clip_pair_by_global_norm(ga_raw, gb_raw, cfg.grad_clip);

                    if grad_norm.is_finite() {
                        grad_norm_sum += grad_norm;
                        grad_norm_count += 1;
                    }
                    if was_clipped {
                        clip_events += 1;
                        step_clip_events += 1;
                    }

                    apply_optimizer_step(
                        &cfg,
                        &mut optim_state,
                        &mut lora_a,
                        &mut lora_b,
                        &ga,
                        &gb,
                        current_lr,
                    );
                    lora_a.zero_grad();
                    lora_b.zero_grad();

                    total_loss += lv;
                    step_loss += lv;
                    loss_count += 1;
                    step_updates += 1;
                    updates_applied += 1;
                    update_cursor += 1;
                    lr_sum += current_lr;
                    step_lr_sum += current_lr;
                    lr_min_seen = lr_min_seen.min(current_lr);
                    lr_max_seen = lr_max_seen.max(current_lr);
                }

                shard_states[shard_idx].update_from_device(lora_a, lora_b, optim_state);
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
            let step_avg_loss = if step_updates > 0 {
                step_loss / step_updates as f64
            } else {
                f64::NAN
            };
            let step_avg_lr = if step_updates > 0 {
                step_lr_sum / step_updates as f64
            } else {
                f64::NAN
            };
            println!(
                "  step {:>4} | loss={:.6} | lr={:.7} | updates={} | clipped={} | time={:.3}s | vram={:.0}MB peak={:.0}MB | frag={:.6}",
                step + 1,
                step_avg_loss,
                step_avg_lr,
                step_updates,
                step_clip_events,
                dt,
                s.allocated as f64 / 1e6,
                s.peak as f64 / 1e6,
                s.frag
            );
        }

        if cfg.save_every > 0 && (step + 1) % cfg.save_every == 0 {
            if let Some(path) = &cfg.checkpoint_path {
                let avg = if loss_count > 0 {
                    total_loss / loss_count as f64
                } else {
                    f64::NAN
                };
                save_training_state(&cfg, path, step + 1, avg, &shard_states)?;
                println!("  checkpoint saved: {} (step {})", path.display(), step + 1);
            }
        }
    }

    let wall = t_total.elapsed().as_secs_f64();
    let sf = snap(&runtime);
    let avg_step = if step_times.is_empty() {
        f64::NAN
    } else {
        step_times.iter().sum::<f64>() / step_times.len() as f64
    };
    let bytes_per_step: usize = shard_bytes_vec.iter().sum();
    let logical_streamed_gb = (bytes_per_step * executed_steps) as f64
        / (1024.0 * 1024.0 * 1024.0);
    let avg_lr = if updates_applied > 0 {
        lr_sum / updates_applied as f64
    } else {
        f64::NAN
    };
    let avg_grad_norm = if grad_norm_count > 0 {
        grad_norm_sum / grad_norm_count as f64
    } else {
        f64::NAN
    };
    let final_lr = if updates_applied > 0 {
        lr_at_update(&cfg, update_cursor.saturating_sub(1), total_updates, warmup_steps)
    } else {
        cfg.lr
    };
    if updates_applied == 0 {
        lr_min_seen = f64::NAN;
        lr_max_seen = f64::NAN;
    }

    if let Some(path) = &cfg.checkpoint_path {
        let avg = if loss_count > 0 {
            total_loss / loss_count as f64
        } else {
            f64::NAN
        };
        save_training_state(&cfg, path, cfg.steps, avg, &shard_states)?;
        println!("  final checkpoint:    {}", path.display());
    }

    println!("\n  Fine-tune engine results:");
    println!("    Logical streamed:  {:.2} GB", logical_streamed_gb);
    println!("    Wall time:         {:.2}s", wall);
    println!("    Avg step:          {:.3}s", avg_step);
    println!("    Updates:           {}", updates_applied);
    println!("    Executed steps:    {}", executed_steps);
    println!("    Avg LR:            {:.8}", avg_lr);
    println!("    Final LR:          {:.8}", final_lr);
    println!("    Avg grad norm:     {:.5}", avg_grad_norm);
    println!("    Clip events:       {}", clip_events);
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
    println!("RESULT mode=streamed_finetune_phase3");
    println!("RESULT weights_source={}", cfg.weights_source.as_str());
    println!("RESULT optimizer={}", cfg.optimizer.as_str());
    println!("RESULT schedule={}", cfg.schedule.as_str());
    println!(
        "RESULT checkpoint_path={}",
        cfg.checkpoint_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!("RESULT checkpoint_version={}", checkpoint_version_runtime);
    println!("RESULT resume={}", if resumed { 1 } else { 0 });
    println!("RESULT restored_optimizer_states={}", restored_optim_states);
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
    println!("RESULT base_lr={:.9}", cfg.lr);
    println!("RESULT min_lr={:.9}", cfg.min_lr);
    println!("RESULT warmup_updates={}", warmup_steps);
    println!("RESULT total_updates={}", total_updates);
    println!("RESULT update_cursor_start={}", initial_update_cursor);
    println!("RESULT update_cursor_final={}", update_cursor);
    println!("RESULT start_step={}", start_step);
    println!("RESULT executed_steps={}", executed_steps);
    println!("RESULT updates_applied={}", updates_applied);
    println!("RESULT inner_steps={}", cfg.inner_steps);
    println!("RESULT grad_clip={:.6}", cfg.grad_clip);
    println!("RESULT weight_decay={:.9}", cfg.weight_decay);
    println!("RESULT avg_lr={:.9}", avg_lr);
    println!("RESULT final_lr={:.9}", final_lr);
    println!("RESULT lr_min_seen={:.9}", lr_min_seen);
    println!("RESULT lr_max_seen={:.9}", lr_max_seen);
    println!("RESULT avg_grad_norm={:.9}", avg_grad_norm);
    println!("RESULT clip_events={}", clip_events);
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
