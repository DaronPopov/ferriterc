use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use ferrite_testkit::env::resolve_repo_root;
use ferrite_testkit::report::{scenario_error, write_json, MatrixResult};
use ferrite_testkit::{run_daemon_smoke, run_scenario_file, RunOptions};

fn main() {
    if let Err(e) = run() {
        eprintln!("ferrite-testkit error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        return Err(anyhow!("missing subcommand"));
    }

    let sub = args.remove(0);
    match sub.as_str() {
        "run" => run_cmd(&args),
        "matrix" => matrix_cmd(&args),
        "daemon-smoke" => daemon_smoke_cmd(&args),
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => Err(anyhow!("unknown subcommand: {other}")),
    }
}

fn run_cmd(args: &[String]) -> Result<()> {
    let scenario = get_flag_value(args, "--scenario")
        .ok_or_else(|| anyhow!("--scenario is required"))?;
    let repo_root = resolve_repo_root(get_flag_value(args, "--repo-root").map(PathBuf::from))?;
    let report_path = get_flag_value(args, "--report").map(PathBuf::from);
    let strict = has_flag(args, "--strict");

    let result = run_scenario_file(
        Path::new(&scenario),
        &repo_root,
        &RunOptions {
            strict_override: if strict { Some(true) } else { None },
        },
    )
    .with_context(|| format!("running scenario {scenario}"))?;

    print_scenario_result(&result);

    if let Some(path) = report_path {
        write_json(&path, &result)?;
        println!("report written: {}", path.display());
    }

    if !result.passed {
        return Err(anyhow!("scenario failed: {}", result.name));
    }
    Ok(())
}

fn matrix_cmd(args: &[String]) -> Result<()> {
    let dir = get_flag_value(args, "--dir").ok_or_else(|| anyhow!("--dir is required"))?;
    let repo_root = resolve_repo_root(get_flag_value(args, "--repo-root").map(PathBuf::from))?;
    let report_path = get_flag_value(args, "--report").map(PathBuf::from);
    let strict = has_flag(args, "--strict");

    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .with_context(|| format!("read scenario dir {dir}"))?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("toml"))
        .collect();
    files.sort();

    let mut scenarios = Vec::new();
    for path in files {
        let display = path.display().to_string();
        match run_scenario_file(
            &path,
            &repo_root,
            &RunOptions {
                strict_override: if strict { Some(true) } else { None },
            },
        ) {
            Ok(res) => {
                print_scenario_result(&res);
                scenarios.push(res);
            }
            Err(e) => {
                eprintln!("[FAIL] {} => {e:#}", display);
                scenarios.push(scenario_error(display, format!("{e:#}")));
            }
        }
    }

    let total = scenarios.len();
    let passed = scenarios.iter().filter(|s| s.passed).count();
    let failed = total.saturating_sub(passed);

    let matrix = MatrixResult {
        total,
        passed,
        failed,
        scenarios,
    };

    println!("matrix summary: {}/{} passed", matrix.passed, matrix.total);

    if let Some(path) = report_path {
        write_json(&path, &matrix)?;
        println!("report written: {}", path.display());
    }

    if matrix.failed > 0 {
        return Err(anyhow!("matrix failed: {} scenario(s)", matrix.failed));
    }
    Ok(())
}

fn daemon_smoke_cmd(args: &[String]) -> Result<()> {
    let repo_root = resolve_repo_root(get_flag_value(args, "--repo-root").map(PathBuf::from))?;
    let report_path = get_flag_value(args, "--report").map(PathBuf::from);

    let result = run_daemon_smoke(&repo_root)?;
    print_scenario_result(&result);

    if let Some(path) = report_path {
        write_json(&path, &result)?;
        println!("report written: {}", path.display());
    }

    if !result.passed {
        return Err(anyhow!("daemon-smoke failed"));
    }
    Ok(())
}

fn print_scenario_result(result: &ferrite_testkit::ScenarioResult) {
    let state = if result.passed { "PASS" } else { "FAIL" };
    println!("[{state}] {} ({} ms)", result.name, result.elapsed_ms);

    for (idx, step) in result.steps.iter().enumerate() {
        let step_state = if step.passed { "PASS" } else { "FAIL" };
        match &step.send {
            Some(send) => println!("  step[{idx}] [{step_state}] {send}"),
            None => println!("  step[{idx}] [{step_state}] {}", step.kind),
        }
    }

    for inv in &result.invariants {
        let inv_state = if inv.passed { "PASS" } else { "FAIL" };
        println!("  inv [{inv_state}] {} => {}", inv.kind, inv.note);
    }
}

fn get_flag_value(args: &[String], flag: &str) -> Option<String> {
    let idx = args.iter().position(|a| a == flag)?;
    args.get(idx + 1).cloned()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn print_usage() {
    println!("ferrite-testkit <subcommand> [options]");
    println!();
    println!("Subcommands:");
    println!("  run --scenario <file> [--repo-root <path>] [--report <file>] [--strict]");
    println!("  matrix --dir <scenarios-dir> [--repo-root <path>] [--report <file>] [--strict]");
    println!("  daemon-smoke [--repo-root <path>] [--report <file>]");
}
