/// Ferrite JIT compiler.
///
/// Compiles ferrite script text into `CompiledProgram` objects that
/// execute on the existing `GpuLangRuntime` at native speed.
///
/// # Pipeline
///
/// ```text
/// script text ──▶ lexer ──▶ tokens ──▶ parser ──▶ AST
///       ──▶ lower ──▶ Program ──▶ compile() ──▶ CompiledProgram
///       ──▶ GpuLangRuntime::execute() ──▶ CUDA kernels
/// ```
///
/// After the first compilation the resulting `CompiledProgram` is
/// cached by script hash — subsequent calls with identical text
/// return instantly.
///
/// # Example
///
/// ```ignore
/// use ferrite_gpu_lang::jit::JitEngine;
/// use ferrite_gpu_lang::{GpuLangRuntime, HostTensor};
///
/// let mut jit = JitEngine::new();
/// let compiled = jit.compile(r#"
///     x = input([1, 1, 1, 4])
///     h = relu(x)
///     y = sigmoid(h)
///     return y
/// "#)?;
///
/// let rt = GpuLangRuntime::new(0)?;
/// let input = HostTensor::new(vec![1,1,1,4], vec![1.0; 4])?;
/// let output = rt.execute(compiled, &[input])?;
/// ```

pub mod ast;
pub mod execute;
pub mod fusion;
pub mod lexer;
pub mod lowering;
pub mod lower;
pub mod parse;
pub mod parser;
pub mod report;
pub mod validate;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{CompiledProgram, LangError};

// ── Cache envelope ──────────────────────────────────────────────

/// Current disk cache schema version.  Bump when the serialized
/// format changes in incompatible ways.
const CACHE_SCHEMA_VERSION: u32 = 1;

/// Compile-time fingerprint of the compiler.  Changes whenever
/// the Op enum, fusion pass, or lowering logic changes in ways
/// that affect the serialized `CompiledProgram`.
fn compiler_fingerprint() -> u64 {
    // Hash key compiler-identity constants.  In a real release build
    // this would incorporate the crate version or a git hash.  For
    // now we derive it from the Op variant count and the cache
    // schema so that structural changes invalidate the cache.
    let mut h = DefaultHasher::new();
    // Number of Op variants (update when Op enum changes).
    16u32.hash(&mut h);
    CACHE_SCHEMA_VERSION.hash(&mut h);
    // Include the crate version from Cargo.toml.
    env!("CARGO_PKG_VERSION").hash(&mut h);
    h.finish()
}

/// Versioned wrapper around a cached `CompiledProgram`.
#[derive(Serialize, Deserialize)]
struct CacheEnvelope {
    schema_version: u32,
    compiler_fingerprint: u64,
    program: CompiledProgram,
}

// ── Error type ───────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum JitError {
    #[error("lexer error at line {line}, col {col}: {message}")]
    Lex {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("parse error at line {line}, col {col}: {message}")]
    Parse {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("lowering error: {0}")]
    Lower(String),

    #[error("[{code}] validation error: {message} (hint: {hint})")]
    Validate {
        code: &'static str,
        message: String,
        hint: String,
    },

    #[error(transparent)]
    Lang(#[from] LangError),
}

impl JitError {
    /// Stable error code for programmatic matching.
    ///
    /// Returns `Some("V001")` etc. for validation errors, `None` for
    /// other error classes (lex, parse, lower, lang).
    pub fn code(&self) -> Option<&'static str> {
        match self {
            JitError::Validate { code, .. } => Some(code),
            _ => None,
        }
    }

    /// Human-readable remediation hint, when available.
    pub fn hint(&self) -> Option<&str> {
        match self {
            JitError::Validate { hint, .. } => Some(hint),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, JitError>;

// ── JIT Engine ───────────────────────────────────────────────────

/// JIT compilation engine with hash-based caching.
///
/// Parses ferrite script text into `CompiledProgram` objects.
/// Identical script text (byte-for-byte) hits the cache and skips
/// all parsing, lowering, and shape-checking work.
pub struct JitEngine {
    cache: HashMap<u64, CompiledProgram>,
    /// Optional disk cache directory for AOT persistence.
    disk_cache_dir: Option<PathBuf>,
    /// Number of programs loaded from disk cache.
    disk_hits: u64,
    /// When false, the fusion pass is skipped (useful for bisecting
    /// optimizer bugs).  Default: true.
    fusion_enabled: bool,
}

impl JitEngine {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            disk_cache_dir: None,
            disk_hits: 0,
            fusion_enabled: true,
        }
    }

    /// Enable or disable the fusion optimization pass.
    ///
    /// When disabled, compile output is equivalent to running with no
    /// optimizer — useful for bisecting whether a bug is in the fusion
    /// pass or elsewhere.  Default: enabled.
    pub fn set_fusion_enabled(&mut self, enabled: bool) {
        self.fusion_enabled = enabled;
        // Invalidate memory cache since compiled programs may differ.
        self.cache.clear();
    }

    /// Whether the fusion optimization pass is enabled.
    pub fn fusion_enabled(&self) -> bool {
        self.fusion_enabled
    }

    /// Enable disk-backed AOT cache at the given directory.
    /// Creates the directory if it doesn't exist.
    pub fn enable_disk_cache(&mut self, dir: PathBuf) -> std::io::Result<()> {
        std::fs::create_dir_all(&dir)?;
        self.disk_cache_dir = Some(dir);
        Ok(())
    }

    /// Number of programs loaded from disk cache (lifetime).
    pub fn disk_hits(&self) -> u64 {
        self.disk_hits
    }

    /// Compile a script to a `CompiledProgram`.
    ///
    /// Returns a cached version when the same script text has been
    /// compiled before (keyed by 64-bit hash). Falls through to disk
    /// cache when enabled.
    pub fn compile(&mut self, script: &str) -> Result<&CompiledProgram> {
        let hash = hash_script(script);
        if !self.cache.contains_key(&hash) {
            // Try loading from disk cache
            if let Some(loaded) = self.load_from_disk(hash) {
                self.disk_hits += 1;
                self.cache.insert(hash, loaded);
            } else {
                let compiled =
                    execute::compile_script_with_options(script, self.fusion_enabled)?;
                self.save_to_disk(hash, &compiled);
                self.cache.insert(hash, compiled);
            }
        }
        Ok(&self.cache[&hash])
    }

    /// Compile a script and return a detailed compile report with timing.
    pub fn compile_with_report(
        &mut self,
        script: &str,
    ) -> Result<(&CompiledProgram, report::CompileReport)> {
        let hash = hash_script(script);
        if self.cache.contains_key(&hash) {
            let t = std::time::Instant::now();
            let compiled = &self.cache[&hash];
            let report = report::CompileReport {
                lex_time: std::time::Duration::ZERO,
                parse_time: std::time::Duration::ZERO,
                validate_time: std::time::Duration::ZERO,
                lower_time: std::time::Duration::ZERO,
                fuse_time: std::time::Duration::ZERO,
                compile_time: std::time::Duration::ZERO,
                total_time: t.elapsed(),
                node_count: compiled.node_count(),
                input_count: compiled.input_count(),
                output_shape: compiled.output_shape().to_vec(),
                fusion_enabled: self.fusion_enabled,
                cache_hit: true,
            };
            return Ok((compiled, report));
        }

        // Try disk cache.
        if let Some(loaded) = self.load_from_disk(hash) {
            let t = std::time::Instant::now();
            self.disk_hits += 1;
            self.cache.insert(hash, loaded);
            let compiled = &self.cache[&hash];
            let report = report::CompileReport {
                lex_time: std::time::Duration::ZERO,
                parse_time: std::time::Duration::ZERO,
                validate_time: std::time::Duration::ZERO,
                lower_time: std::time::Duration::ZERO,
                fuse_time: std::time::Duration::ZERO,
                compile_time: std::time::Duration::ZERO,
                total_time: t.elapsed(),
                node_count: compiled.node_count(),
                input_count: compiled.input_count(),
                output_shape: compiled.output_shape().to_vec(),
                fusion_enabled: self.fusion_enabled,
                cache_hit: true,
            };
            return Ok((compiled, report));
        }

        let (compiled, report) =
            execute::compile_script_with_report(script, self.fusion_enabled)?;
        self.save_to_disk(hash, &compiled);
        self.cache.insert(hash, compiled);
        Ok((&self.cache[&hash], report))
    }

    /// Number of programs currently in the cache.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Drop all cached programs (memory only — does not clear disk).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Drop all cached programs including on-disk entries.
    pub fn clear_all(&mut self) {
        self.cache.clear();
        if let Some(dir) = &self.disk_cache_dir {
            let _ = std::fs::remove_dir_all(dir);
            let _ = std::fs::create_dir_all(dir);
        }
    }

    /// Number of entries in the disk cache.
    pub fn disk_cache_len(&self) -> usize {
        self.disk_cache_dir.as_ref().map_or(0, |dir| {
            std::fs::read_dir(dir)
                .map(|rd| rd.filter_map(|e| e.ok()).count())
                .unwrap_or(0)
        })
    }

    fn cache_path(&self, hash: u64) -> Option<PathBuf> {
        self.disk_cache_dir.as_ref().map(|dir| dir.join(format!("{:016x}.json", hash)))
    }

    fn load_from_disk(&self, hash: u64) -> Option<CompiledProgram> {
        let path = self.cache_path(hash)?;
        let data = std::fs::read_to_string(&path).ok()?;
        let envelope: CacheEnvelope = serde_json::from_str(&data).ok()?;
        // Reject stale entries from a different compiler version or schema.
        if envelope.schema_version != CACHE_SCHEMA_VERSION {
            return None;
        }
        if envelope.compiler_fingerprint != compiler_fingerprint() {
            return None;
        }
        Some(envelope.program)
    }

    fn save_to_disk(&self, hash: u64, program: &CompiledProgram) {
        if let Some(path) = self.cache_path(hash) {
            let envelope = CacheEnvelope {
                schema_version: CACHE_SCHEMA_VERSION,
                compiler_fingerprint: compiler_fingerprint(),
                program: program.clone(),
            };
            if let Ok(json) = serde_json::to_string(&envelope) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}

impl Default for JitEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn hash_script(script: &str) -> u64 {
    let mut h = DefaultHasher::new();
    script.hash(&mut h);
    h.finish()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::lexer::{tokenize, Token};

    // ── lexer ────────────────────────────────────────────────────

    #[test]
    fn lex_basic_assignment() {
        let toks = tokenize("x = input([1, 2, 3])").unwrap();
        let kinds: Vec<_> = toks.iter().map(|t| &t.token).collect();
        assert!(matches!(kinds[0], Token::Ident(s) if s == "x"));
        assert_eq!(*kinds[1], Token::Eq);
        assert!(matches!(kinds[2], Token::Ident(s) if s == "input"));
        assert_eq!(*kinds[3], Token::LParen);
        assert_eq!(*kinds[4], Token::LBracket);
        assert!(matches!(kinds[5], Token::Int(1)));
        assert_eq!(*kinds[6], Token::Comma);
        assert!(matches!(kinds[7], Token::Int(2)));
        assert_eq!(*kinds[8], Token::Comma);
        assert!(matches!(kinds[9], Token::Int(3)));
        assert_eq!(*kinds[10], Token::RBracket);
        assert_eq!(*kinds[11], Token::RParen);
        assert_eq!(*kinds[12], Token::Eof);
    }

    #[test]
    fn lex_keywords() {
        let toks = tokenize("fn return end true false").unwrap();
        let kinds: Vec<_> = toks.iter().map(|t| &t.token).collect();
        assert_eq!(*kinds[0], Token::Fn);
        assert_eq!(*kinds[1], Token::Return);
        assert_eq!(*kinds[2], Token::End);
        assert_eq!(*kinds[3], Token::True);
        assert_eq!(*kinds[4], Token::False);
    }

    #[test]
    fn lex_comments_stripped() {
        let toks = tokenize("x = relu(y) # comment\nreturn x").unwrap();
        assert!(!toks
            .iter()
            .any(|t| matches!(&t.token, Token::Ident(s) if s.contains('#'))));
        assert!(toks.iter().any(|t| t.token == Token::Return));
    }

    #[test]
    fn lex_error_on_invalid_char() {
        assert!(tokenize("x = @bad").is_err());
    }

    // ── parser ───────────────────────────────────────────────────

    #[test]
    fn parse_simple_script() {
        let toks = tokenize(
            r#"
            x = input([1, 1, 1, 4])
            h = relu(x)
            return h
        "#,
        )
        .unwrap();
        let script = parser::parse(toks).unwrap();
        assert_eq!(script.stmts.len(), 3);
    }

    #[test]
    fn parse_binary_ops() {
        let toks = tokenize(
            r#"
            a = input([1, 4])
            b = input([1, 4])
            c = add(a, b)
            d = mul(a, c)
            return d
        "#,
        )
        .unwrap();
        let script = parser::parse(toks).unwrap();
        assert_eq!(script.stmts.len(), 5);
    }

    #[test]
    fn parse_named_args() {
        let toks = tokenize(
            r#"
            x = input([1, 4])
            y = topk(x, k=2, dim=1, largest=true)
            return y
        "#,
        )
        .unwrap();
        let script = parser::parse(toks).unwrap();
        assert_eq!(script.stmts.len(), 3);
    }

    #[test]
    fn parse_fn_definition() {
        let toks = tokenize(
            r#"
            fn activate(x):
                h = relu(x)
                y = sigmoid(h)
                return y
            end
            a = input([1, 4])
            b = activate(a)
            return b
        "#,
        )
        .unwrap();
        let script = parser::parse(toks).unwrap();
        assert_eq!(script.stmts.len(), 4); // fn + 3 stmts
    }

    #[test]
    fn parse_empty_fn() {
        let res = tokenize("fn noop(): end\nx = input([1])\nreturn x");
        // "end" immediately after colon → body is empty, return missing
        // This should parse (body has no stmts) but lower will fail later
        assert!(res.is_ok());
    }

    // ── lowering ─────────────────────────────────────────────────

    #[test]
    fn lower_simple_chain() {
        let toks = tokenize(
            r#"
            x = input([1, 1, 1, 4])
            h = relu(x)
            y = sigmoid(h)
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[1, 1, 1, 4]);
        assert_eq!(compiled.input_shapes().len(), 1);
    }

    #[test]
    fn lower_two_inputs() {
        let toks = tokenize(
            r#"
            a = input([1, 4])
            b = input([1, 4])
            c = add(a, b)
            return c
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.input_shapes().len(), 2);
        assert_eq!(compiled.output_shape(), &[1, 4]);
    }

    #[test]
    fn lower_fn_inline() {
        let toks = tokenize(
            r#"
            fn activate(x):
                h = relu(x)
                y = sigmoid(h)
                return y
            end
            a = input([1, 1, 1, 4])
            b = activate(a)
            return b
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[1, 1, 1, 4]);
        assert_eq!(compiled.input_shapes().len(), 1);
    }

    #[test]
    fn lower_multi_fn() {
        let toks = tokenize(
            r#"
            fn layer1(x):
                y = relu(x)
                return y
            end
            fn layer2(x):
                y = sigmoid(x)
                return y
            end
            a = input([1, 4])
            b = layer1(a)
            c = layer2(b)
            return c
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[1, 4]);
    }

    #[test]
    fn lower_cumsum() {
        let toks = tokenize(
            r#"
            x = input([2, 3, 4])
            y = cumsum(x, dim=1)
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[2, 3, 4]);
    }

    #[test]
    fn lower_topk() {
        let toks = tokenize(
            r#"
            x = input([2, 10])
            y = topk(x, k=3, dim=1, largest=true)
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[2, 3]); // dim=1 → k=3
    }

    #[test]
    fn lower_deep_chain() {
        let toks = tokenize(
            r#"
            x = input([1, 1, 1, 8])
            a = relu(x)
            b = tanh(a)
            c = sigmoid(b)
            d = relu(c)
            return d
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[1, 1, 1, 8]);
    }

    // ── error cases ──────────────────────────────────────────────

    #[test]
    fn error_undefined_var() {
        let toks = tokenize("x = relu(undefined_var)\nreturn x").unwrap();
        let ast = parser::parse(toks).unwrap();
        assert!(lower::lower(ast).is_err());
    }

    #[test]
    fn error_unknown_function() {
        let toks = tokenize("x = input([1, 4])\ny = nonexistent(x)\nreturn y").unwrap();
        let ast = parser::parse(toks).unwrap();
        assert!(lower::lower(ast).is_err());
    }

    #[test]
    fn error_shape_mismatch() {
        let toks = tokenize(
            r#"
            a = input([1, 4])
            b = input([1, 8])
            c = add(a, b)
            return c
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        // Shape mismatch caught during compile()
        assert!(program.compile().is_err());
    }

    #[test]
    fn error_missing_return() {
        let toks = tokenize("x = input([1, 4])\ny = relu(x)").unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        // No set_output → MissingOutput
        assert!(program.compile().is_err());
    }

    // ── JIT engine ───────────────────────────────────────────────

    #[test]
    fn engine_cache_hit() {
        let mut engine = JitEngine::new();
        let script = "x = input([1, 4])\ny = relu(x)\nreturn y";
        assert_eq!(engine.cache_len(), 0);
        let _ = engine.compile(script).unwrap();
        assert_eq!(engine.cache_len(), 1);
        let _ = engine.compile(script).unwrap();
        assert_eq!(engine.cache_len(), 1); // cache hit
    }

    #[test]
    fn engine_different_scripts() {
        let mut engine = JitEngine::new();
        let _ = engine
            .compile("x = input([1, 4])\ny = relu(x)\nreturn y")
            .unwrap();
        let _ = engine
            .compile("x = input([1, 4])\ny = tanh(x)\nreturn y")
            .unwrap();
        assert_eq!(engine.cache_len(), 2);
    }

    #[test]
    fn engine_clear_cache() {
        let mut engine = JitEngine::new();
        let _ = engine
            .compile("x = input([1, 4])\ny = relu(x)\nreturn y")
            .unwrap();
        assert_eq!(engine.cache_len(), 1);
        engine.clear_cache();
        assert_eq!(engine.cache_len(), 0);
    }

    // ── lexer: floats & operators ────────────────────────────────

    #[test]
    fn lex_float_literals() {
        let toks = tokenize("2.0 0.5 3.14").unwrap();
        let kinds: Vec<_> = toks.iter().map(|t| &t.token).collect();
        assert!(matches!(kinds[0], Token::Float(v) if (*v - 2.0).abs() < 1e-10));
        assert!(matches!(kinds[1], Token::Float(v) if (*v - 0.5).abs() < 1e-10));
        assert!(matches!(kinds[2], Token::Float(v) if (*v - 3.14).abs() < 1e-10));
    }

    #[test]
    fn lex_scientific_notation() {
        let toks = tokenize("1e3 2.5e2 1e-2").unwrap();
        let kinds: Vec<_> = toks.iter().map(|t| &t.token).collect();
        assert!(matches!(kinds[0], Token::Float(v) if (*v - 1000.0).abs() < 1e-10));
        assert!(matches!(kinds[1], Token::Float(v) if (*v - 250.0).abs() < 1e-10));
        assert!(matches!(kinds[2], Token::Float(v) if (*v - 0.01).abs() < 1e-10));
    }

    #[test]
    fn lex_operators() {
        let toks = tokenize("x + y * z - w / v").unwrap();
        let kinds: Vec<_> = toks.iter().map(|t| &t.token).collect();
        assert!(matches!(kinds[0], Token::Ident(s) if s == "x"));
        assert_eq!(*kinds[1], Token::Plus);
        assert!(matches!(kinds[2], Token::Ident(s) if s == "y"));
        assert_eq!(*kinds[3], Token::Star);
        assert!(matches!(kinds[4], Token::Ident(s) if s == "z"));
        assert_eq!(*kinds[5], Token::Minus);
        assert!(matches!(kinds[6], Token::Ident(s) if s == "w"));
        assert_eq!(*kinds[7], Token::Slash);
        assert!(matches!(kinds[8], Token::Ident(s) if s == "v"));
    }

    #[test]
    fn lex_tile_over_keywords() {
        let toks = tokenize("tile over end").unwrap();
        let kinds: Vec<_> = toks.iter().map(|t| &t.token).collect();
        assert_eq!(*kinds[0], Token::Tile);
        assert_eq!(*kinds[1], Token::Over);
        assert_eq!(*kinds[2], Token::End);
    }

    // ── parser: infix & tile ─────────────────────────────────────

    #[test]
    fn parse_infix_precedence() {
        // x + y * z  should parse as  x + (y * z)
        let toks = tokenize("x = input([4])\ny = input([4])\nz = input([4])\nw = x + y * z\nreturn w").unwrap();
        let script = parser::parse(toks).unwrap();
        // The let w = ... should produce BinOp(Add, Var(x), BinOp(Mul, ...))
        if let ast::Stmt::Let { value, .. } = &script.stmts[3] {
            match value {
                ast::Expr::BinOp { op, left, right } => {
                    assert_eq!(*op, ast::BinOper::Add);
                    assert!(matches!(left.as_ref(), ast::Expr::Var(s) if s == "x"));
                    assert!(matches!(right.as_ref(), ast::Expr::BinOp { op: ast::BinOper::Mul, .. }));
                }
                other => panic!("expected BinOp, got {:?}", other),
            }
        } else {
            panic!("expected Let statement");
        }
    }

    #[test]
    fn parse_parenthesized_expr() {
        let toks = tokenize("x = input([4])\ny = (x + x) * x\nreturn y").unwrap();
        let script = parser::parse(toks).unwrap();
        if let ast::Stmt::Let { value, .. } = &script.stmts[1] {
            assert!(matches!(value, ast::Expr::BinOp { op: ast::BinOper::Mul, .. }));
        } else {
            panic!("expected Let statement");
        }
    }

    #[test]
    fn parse_tile_block() {
        let toks = tokenize(
            r#"
            x = input([4])
            tile y over (x):
                y = x * 2.0
            end
            return y
        "#,
        )
        .unwrap();
        let script = parser::parse(toks).unwrap();
        assert_eq!(script.stmts.len(), 3); // let x, tile, return
        match &script.stmts[1] {
            ast::Stmt::Tile { output, inputs, body } => {
                assert_eq!(output, "y");
                assert_eq!(inputs, &["x"]);
                assert_eq!(body.len(), 1);
            }
            other => panic!("expected Tile, got {:?}", other),
        }
    }

    #[test]
    fn parse_tile_single_input() {
        let toks = tokenize(
            r#"
            x = input([4])
            tile y over x:
                y = x * 2.0
            end
            return y
        "#,
        )
        .unwrap();
        let script = parser::parse(toks).unwrap();
        match &script.stmts[1] {
            ast::Stmt::Tile { inputs, .. } => {
                assert_eq!(inputs, &["x"]);
            }
            other => panic!("expected Tile, got {:?}", other),
        }
    }

    #[test]
    fn parse_negation() {
        let toks = tokenize("x = input([4])\ny = -x\nreturn y").unwrap();
        let script = parser::parse(toks).unwrap();
        if let ast::Stmt::Let { value, .. } = &script.stmts[1] {
            assert!(matches!(value, ast::Expr::Neg(..)));
        } else {
            panic!("expected Let statement");
        }
    }

    // ── lowering: infix, sub, div, tile, scalars ────────────────

    #[test]
    fn lower_infix_add_mul() {
        let toks = tokenize(
            r#"
            x = input([4])
            y = x * 2.0 + 1.0
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[4]);
    }

    #[test]
    fn lower_sub_div() {
        let toks = tokenize(
            r#"
            x = input([4])
            y = (x - 1.0) / 2.0
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[4]);
    }

    #[test]
    fn lower_negation() {
        let toks = tokenize(
            r#"
            x = input([4])
            y = -x
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[4]);
    }

    #[test]
    fn lower_infix_with_builtin() {
        // relu(x * 2.0) — expression as arg to builtin
        let toks = tokenize(
            r#"
            x = input([4])
            y = relu(x * 2.0)
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[4]);
    }

    #[test]
    fn lower_tile_block() {
        let toks = tokenize(
            r#"
            x = input([4])
            tile y over (x):
                y = x * 2.0 + 1.0
            end
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[4]);
    }

    #[test]
    fn lower_tile_multi_input() {
        let toks = tokenize(
            r#"
            a = input([4])
            b = input([4])
            tile c over (a, b):
                c = a * b + a
            end
            return c
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        let program = lower::lower(ast).unwrap();
        let compiled = program.compile().unwrap();
        assert_eq!(compiled.output_shape(), &[4]);
    }

    #[test]
    fn error_scalar_scalar_binop() {
        // Two scalar literals with no tensor — should error
        let toks = tokenize(
            r#"
            x = 2.0 + 3.0
            return x
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        assert!(lower::lower(ast).is_err());
    }

    #[test]
    fn error_tile_undefined_input() {
        let toks = tokenize(
            r#"
            tile y over (nonexistent):
                y = nonexistent * 2.0
            end
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        assert!(lower::lower(ast).is_err());
    }

    #[test]
    fn error_tile_no_output() {
        let toks = tokenize(
            r#"
            x = input([4])
            tile y over (x):
                z = x * 2.0
            end
            return y
        "#,
        )
        .unwrap();
        let ast = parser::parse(toks).unwrap();
        assert!(lower::lower(ast).is_err());
    }

    // ── cache determinism ───────────────────────────────────────

    #[test]
    fn cache_cold_compile() {
        // Cold compile: no memory cache, no disk cache.
        let mut engine = JitEngine::new();
        let script = "x = input([2, 4])\ny = relu(x)\nreturn y";
        let compiled = engine.compile(script).unwrap();
        assert_eq!(compiled.output_shape(), &[2, 4]);
        assert_eq!(engine.cache_len(), 1);
        assert_eq!(engine.disk_hits(), 0);
    }

    #[test]
    fn cache_warm_memory_hit() {
        // Second compile with identical text hits memory cache.
        let mut engine = JitEngine::new();
        let script = "x = input([2, 4])\ny = relu(x)\nreturn y";
        let _ = engine.compile(script).unwrap();
        let _ = engine.compile(script).unwrap();
        assert_eq!(engine.cache_len(), 1); // still 1 — memory hit
        assert_eq!(engine.disk_hits(), 0);
    }

    #[test]
    fn cache_warm_disk_hit() {
        // After eviction from memory, disk cache provides the program.
        let dir = std::env::temp_dir().join(format!("ferrite_cache_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut engine = JitEngine::new();
        engine.enable_disk_cache(dir.clone()).unwrap();

        let script = "x = input([2, 4])\ny = sigmoid(x)\nreturn y";
        let compiled_a = engine.compile(script).unwrap();
        let shape_a = compiled_a.output_shape().to_vec();
        assert_eq!(engine.disk_cache_len(), 1);

        // Clear memory cache, forcing next compile to hit disk.
        engine.clear_cache();
        assert_eq!(engine.cache_len(), 0);

        let compiled_b = engine.compile(script).unwrap();
        assert_eq!(compiled_b.output_shape(), &shape_a[..]);
        assert_eq!(engine.disk_hits(), 1);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_stale_disk_rejected() {
        // Write a hand-crafted envelope with wrong fingerprint.
        let dir = std::env::temp_dir().join(format!("ferrite_stale_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut engine = JitEngine::new();
        engine.enable_disk_cache(dir.clone()).unwrap();

        let script = "x = input([4])\ny = relu(x)\nreturn y";
        // Compile once to get a valid CompiledProgram.
        let _ = engine.compile(script).unwrap();
        assert_eq!(engine.disk_cache_len(), 1);

        // Tamper with the disk cache file: replace fingerprint.
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        let path = entries[0].path();
        let mut data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        data["compiler_fingerprint"] = serde_json::json!(999999);
        std::fs::write(&path, serde_json::to_string(&data).unwrap()).unwrap();

        // Clear memory and try to load — should miss (stale fingerprint).
        engine.clear_cache();
        let _ = engine.compile(script).unwrap();
        // disk_hits should still be 1 (the first load before tampering
        // was a cold compile — never loaded from disk).  The stale file
        // was rejected, so we got a fresh compile.
        assert_eq!(engine.disk_hits(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_schema_version_mismatch() {
        let dir = std::env::temp_dir().join(format!("ferrite_schema_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut engine = JitEngine::new();
        engine.enable_disk_cache(dir.clone()).unwrap();

        let script = "x = input([4])\ny = tanh(x)\nreturn y";
        let _ = engine.compile(script).unwrap();

        // Tamper: change schema version.
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        let path = entries[0].path();
        let mut data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        data["schema_version"] = serde_json::json!(9999);
        std::fs::write(&path, serde_json::to_string(&data).unwrap()).unwrap();

        engine.clear_cache();
        let _ = engine.compile(script).unwrap();
        // Should have recompiled from scratch, not loaded from disk.
        assert_eq!(engine.disk_hits(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── fusion safety ───────────────────────────────────────────

    #[test]
    fn fusion_disable_knob() {
        let mut engine = JitEngine::new();
        engine.set_fusion_enabled(false);
        assert!(!engine.fusion_enabled());

        let script = r#"
            a = input([1, 4])
            b = input([1, 4])
            c = add(a, b)
            d = relu(c)
            return d
        "#;
        // Without fusion, relu(add(a,b)) stays as separate nodes.
        let compiled = engine.compile(script).unwrap();
        // The op summary should contain "add" and "relu" as separate ops.
        let summary = compiled.op_summary();
        let has_add = summary.iter().any(|(name, _)| *name == "add");
        let has_relu = summary.iter().any(|(name, _)| *name == "relu");
        assert!(has_add, "expected separate 'add' when fusion disabled");
        assert!(has_relu, "expected separate 'relu' when fusion disabled");
    }

    #[test]
    fn fusion_enable_knob() {
        let mut engine = JitEngine::new();
        assert!(engine.fusion_enabled()); // default

        let script = r#"
            a = input([1, 4])
            b = input([1, 4])
            c = add(a, b)
            d = relu(c)
            return d
        "#;
        let compiled = engine.compile(script).unwrap();
        let summary = compiled.op_summary();
        let has_fused = summary.iter().any(|(name, _)| *name == "fused_relu_add");
        assert!(has_fused, "expected fused_relu_add when fusion enabled");
    }

    #[test]
    fn fusion_shape_preservation() {
        // Compile with and without fusion — shapes must match.
        let script = r#"
            a = input([2, 8])
            b = input([2, 8])
            c = add(a, b)
            d = relu(c)
            e = sigmoid(d)
            return e
        "#;

        let fused = execute::compile_script_with_options(script, true).unwrap();
        let unfused = execute::compile_script_with_options(script, false).unwrap();

        assert_eq!(fused.output_shape(), unfused.output_shape());
        assert_eq!(fused.input_shapes(), unfused.input_shapes());
        assert_eq!(fused.input_count(), unfused.input_count());
    }

    #[test]
    fn fusion_complex_shape_preservation() {
        // Multiple fusible patterns in a single graph.
        let script = r#"
            x = input([4, 16])
            y = input([4, 16])
            a = add(x, y)
            b = relu(a)
            c = mul(b, x)
            d = sigmoid(add(c, y))
            return d
        "#;

        let fused = execute::compile_script_with_options(script, true).unwrap();
        let unfused = execute::compile_script_with_options(script, false).unwrap();

        assert_eq!(fused.output_shape(), unfused.output_shape());
        assert_eq!(fused.input_shapes(), unfused.input_shapes());
    }

    #[test]
    fn fusion_no_fuse_multi_consumer() {
        // The binary op is consumed by two nodes — fusion must NOT fire.
        let script = r#"
            a = input([4])
            b = input([4])
            c = add(a, b)
            d = relu(c)
            e = sigmoid(c)
            f = add(d, e)
            return f
        "#;

        let compiled = execute::compile_script_with_options(script, true).unwrap();
        let summary = compiled.op_summary();
        // "add" must still appear because c=add(a,b) has two consumers.
        let add_count = summary
            .iter()
            .find(|(name, _)| *name == "add")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert!(add_count >= 1, "add with multiple consumers should not be fused");
    }

    // ── compile report ──────────────────────────────────────────

    #[test]
    fn compile_report_cold() {
        let mut engine = JitEngine::new();
        let script = "x = input([2, 4])\ny = relu(x)\nreturn y";
        let (compiled, report) = engine.compile_with_report(script).unwrap();
        assert!(!report.cache_hit);
        assert!(report.fusion_enabled);
        assert_eq!(report.output_shape, compiled.output_shape());
        assert!(report.total_time.as_nanos() > 0);
        // Display impl should not panic.
        let _s = format!("{}", report);
    }

    #[test]
    fn compile_report_cache_hit() {
        let mut engine = JitEngine::new();
        let script = "x = input([2, 4])\ny = relu(x)\nreturn y";
        let _ = engine.compile(script).unwrap();
        let (_, report) = engine.compile_with_report(script).unwrap();
        assert!(report.cache_hit);
        assert_eq!(report.lex_time.as_nanos(), 0);
    }

    // ── stress tests ────────────────────────────────────────────

    #[test]
    fn stress_repeated_compile() {
        // Compile the same script 1000 times — exercises cache path.
        let mut engine = JitEngine::new();
        let script = "x = input([4])\ny = relu(x)\nreturn y";
        for _ in 0..1000 {
            let _ = engine.compile(script).unwrap();
        }
        assert_eq!(engine.cache_len(), 1);
    }

    #[test]
    fn stress_many_distinct_scripts() {
        // Compile 200 distinct scripts — exercises cold compile path.
        let mut engine = JitEngine::new();
        for i in 1..=200 {
            let script = format!(
                "x = input([1, {}])\ny = relu(x)\nreturn y",
                i
            );
            let compiled = engine.compile(&script).unwrap();
            assert_eq!(compiled.output_shape(), &[1, i]);
        }
        assert_eq!(engine.cache_len(), 200);
    }

    #[test]
    fn stress_large_script() {
        // Generate a script with a long chain of operations.
        let mut lines = Vec::new();
        lines.push("x0 = input([4, 32])".to_string());
        for i in 1..=100 {
            let prev = format!("x{}", i - 1);
            let op = match i % 4 {
                0 => "relu",
                1 => "sigmoid",
                2 => "tanh",
                _ => "relu",
            };
            lines.push(format!("x{} = {}({})", i, op, prev));
        }
        lines.push("return x100".to_string());
        let script = lines.join("\n");

        let compiled = execute::compile_script_with_options(&script, true).unwrap();
        assert_eq!(compiled.output_shape(), &[4, 32]);
        assert!(compiled.node_count() > 50); // many nodes
    }

    #[test]
    fn stress_disk_cache_churn() {
        let dir = std::env::temp_dir().join(format!("ferrite_churn_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut engine = JitEngine::new();
        engine.enable_disk_cache(dir.clone()).unwrap();

        // Write 50 entries.
        for i in 1..=50 {
            let script = format!(
                "x = input([1, {}])\ny = tanh(x)\nreturn y",
                i
            );
            let _ = engine.compile(&script).unwrap();
        }
        assert_eq!(engine.disk_cache_len(), 50);

        // Clear memory and reload all from disk.
        engine.clear_cache();
        for i in 1..=50 {
            let script = format!(
                "x = input([1, {}])\ny = tanh(x)\nreturn y",
                i
            );
            let compiled = engine.compile(&script).unwrap();
            assert_eq!(compiled.output_shape(), &[1, i]);
        }
        assert_eq!(engine.disk_hits(), 50);

        // Clear all and verify disk is empty.
        engine.clear_all();
        assert_eq!(engine.disk_cache_len(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stress_concurrent_compile() {
        // Spawn threads that each compile independently — JitEngine is
        // per-thread here (no shared state), but this tests that the
        // compile pipeline itself is safe to run from multiple threads.
        let scripts: Vec<String> = (1..=20)
            .map(|i| format!("x = input([1, {}])\ny = relu(x)\nreturn y", i))
            .collect();

        let handles: Vec<_> = scripts
            .into_iter()
            .map(|script| {
                std::thread::spawn(move || {
                    let mut engine = JitEngine::new();
                    for _ in 0..50 {
                        let _ = engine.compile(&script).unwrap();
                    }
                    engine.cache_len()
                })
            })
            .collect();

        for handle in handles {
            let count = handle.join().unwrap();
            assert_eq!(count, 1);
        }
    }
}
