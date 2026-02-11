use std::time::Instant;

use super::fusion;
use super::lowering;
use super::parse;
use super::report::CompileReport;
use super::validate;
use super::Result;
use crate::CompiledProgram;

/// Full parse/lower/compile pipeline facade.
///
/// Pipeline: script → parse → AST → validate → lower → Program → fuse → compile → CompiledProgram
pub fn compile_script(script: &str) -> Result<CompiledProgram> {
    compile_script_with_options(script, true)
}

/// Compile with explicit fusion control (for debugging/bisecting).
pub fn compile_script_with_options(script: &str, enable_fusion: bool) -> Result<CompiledProgram> {
    let ast = parse::parse_script(script)?;
    validate::validate(&ast)?;
    let program = lowering::lower_script(ast)?;
    let compiled = if enable_fusion {
        let fused = fusion::fuse(program);
        fused.compile()?
    } else {
        program.compile()?
    };
    Ok(compiled)
}

/// Compile with full stage-level timing, returning both the program
/// and a [`CompileReport`] for ops/debugging.
pub fn compile_script_with_report(
    script: &str,
    enable_fusion: bool,
) -> Result<(CompiledProgram, CompileReport)> {
    let wall_start = Instant::now();

    let t0 = Instant::now();
    let tokens = super::lexer::tokenize(script)?;
    let lex_time = t0.elapsed();

    let t0 = Instant::now();
    let ast = super::parser::parse(tokens)?;
    let parse_time = t0.elapsed();

    let t0 = Instant::now();
    validate::validate(&ast)?;
    let validate_time = t0.elapsed();

    let t0 = Instant::now();
    let program = super::lower::lower(ast)?;
    let lower_time = t0.elapsed();

    let t0 = Instant::now();
    let program = if enable_fusion {
        fusion::fuse(program)
    } else {
        program
    };
    let fuse_time = t0.elapsed();

    let t0 = Instant::now();
    let compiled = program.compile()?;
    let compile_time = t0.elapsed();

    let total_time = wall_start.elapsed();

    let report = CompileReport {
        lex_time,
        parse_time,
        validate_time,
        lower_time,
        fuse_time,
        compile_time,
        total_time,
        node_count: compiled.node_count(),
        input_count: compiled.input_count(),
        output_shape: compiled.output_shape().to_vec(),
        fusion_enabled: enable_fusion,
        cache_hit: false,
    };

    Ok((compiled, report))
}
