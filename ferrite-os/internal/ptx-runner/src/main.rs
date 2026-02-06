use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TargetKind {
    Example,
    Bin,
    Script,
}

#[derive(Debug, Clone)]
struct Target {
    name: String,
    crate_name: Option<String>,
    kind: TargetKind,
    path: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ptx-runner error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();

    let workspace_root = find_workspace_root(env::current_dir().map_err(|e| e.to_string())?)
        .ok_or_else(|| "Unable to locate workspace root (Cargo.toml with [workspace])".to_string())?;
    let repo_root = workspace_root.parent().unwrap_or(&workspace_root).to_path_buf();

    let targets = discover_targets(&workspace_root, &repo_root)?;

    if args.is_empty() || args[0] == "list" || args[0] == "--list" || args[0] == "-l" {
        print_targets(&targets);
        return Ok(());
    }

    let (target_key, passthrough) = (args[0].as_str(), &args[1..]);
    let (crate_hint, name) = parse_target_hint(target_key);

    let matches = resolve_target(&targets, crate_hint.as_deref(), name);
    if matches.is_empty() {
        print_targets(&targets);
        return Err(format!("Unknown target '{target_key}'"));
    }
    if matches.len() > 1 {
        eprintln!("Ambiguous target '{target_key}'. Please disambiguate with crate:name.");
        for t in matches {
            eprintln!("  {}:{} ({:?})", t.crate_name.clone().unwrap_or_default(), t.name, t.kind);
        }
        return Err("Ambiguous target".to_string());
    }

    let target = &matches[0];
    match target.kind {
        TargetKind::Example => run_cargo_example(&workspace_root, &repo_root, target, passthrough)?,
        TargetKind::Bin => run_cargo_bin(&workspace_root, &repo_root, target, passthrough)?,
        TargetKind::Script => run_script(&workspace_root, &repo_root, target, passthrough)?,
    }

    Ok(())
}

fn parse_target_hint(input: &str) -> (Option<String>, String) {
    if let Some((a, b)) = input.split_once("::") {
        return (Some(a.to_string()), b.to_string());
    }
    if let Some((a, b)) = input.split_once(':') {
        return (Some(a.to_string()), b.to_string());
    }
    if let Some((a, b)) = input.split_once('/') {
        return (Some(a.to_string()), b.to_string());
    }
    (None, input.to_string())
}

fn resolve_target<'a>(targets: &'a [Target], crate_hint: Option<&str>, name: String) -> Vec<&'a Target> {
    targets
        .iter()
        .filter(|t| t.name == name)
        .filter(|t| match crate_hint {
            Some(hint) => t.crate_name.as_deref() == Some(hint),
            None => true,
        })
        .collect()
}

fn print_targets(targets: &[Target]) {
    println!("Available targets:");
    for kind in [TargetKind::Example, TargetKind::Bin, TargetKind::Script] {
        let mut list: Vec<&Target> = targets.iter().filter(|t| t.kind == kind).collect();
        if !list.is_empty() {
            println!("\n{:?}:", kind);
            list.sort_by_key(|t| (t.crate_name.clone().unwrap_or_default(), t.name.clone()));
            for t in list {
                let crate_name = t.crate_name.clone().unwrap_or_default();
                if crate_name.is_empty() {
                    println!("  {}", t.name);
                } else {
                    println!("  {}:{}", crate_name, t.name);
                }
            }
        }
    }

    println!("\nUsage:");
    println!("  cargo run -- <target> [args...]");
    println!("  cargo run -- list");
}

fn run_cargo_example(
    workspace_root: &Path,
    repo_root: &Path,
    target: &Target,
    args: &[String],
) -> Result<(), String> {
    let crate_name = target.crate_name.clone().ok_or_else(|| "Missing crate for example".to_string())?;

    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace_root)
        .arg("run")
        .arg("-p")
        .arg(crate_name)
        .arg("--example")
        .arg(&target.name);

    if !args.is_empty() {
        cmd.arg("--");
        cmd.args(args);
    }

    apply_ld_library_path(&mut cmd, repo_root);
    run_command(cmd)
}

fn run_cargo_bin(
    workspace_root: &Path,
    repo_root: &Path,
    target: &Target,
    args: &[String],
) -> Result<(), String> {
    let crate_name = target.crate_name.clone().ok_or_else(|| "Missing crate for bin".to_string())?;

    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace_root)
        .arg("run")
        .arg("-p")
        .arg(crate_name)
        .arg("--bin")
        .arg(&target.name);

    if !args.is_empty() {
        cmd.arg("--");
        cmd.args(args);
    }

    apply_ld_library_path(&mut cmd, repo_root);
    run_command(cmd)
}

fn run_script(
    workspace_root: &Path,
    repo_root: &Path,
    target: &Target,
    args: &[String],
) -> Result<(), String> {
    let path = target.path.as_ref().ok_or_else(|| "Missing script path".to_string())?;

    let mut cmd = Command::new("bash");
    cmd.current_dir(workspace_root)
        .arg(path);
    if !args.is_empty() {
        cmd.args(args);
    }

    apply_ld_library_path(&mut cmd, repo_root);
    run_command(cmd)
}

fn run_command(mut cmd: Command) -> Result<(), String> {
    let status = cmd.status().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Command failed with status {status}"))
    }
}

fn apply_ld_library_path(cmd: &mut Command, repo_root: &Path) {
    let lib_dir = repo_root.join("lib");
    if !lib_dir.exists() {
        return;
    }
    let lib_str = lib_dir.to_string_lossy().to_string();
    let existing = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if existing.split(':').any(|p| p == lib_str) {
        return;
    }
    let new_val = if existing.is_empty() {
        lib_str
    } else {
        format!("{}:{}", lib_str, existing)
    };
    cmd.env("LD_LIBRARY_PATH", new_val);
}

fn find_workspace_root(start: PathBuf) -> Option<PathBuf> {
    let mut cur = start;
    loop {
        let candidate = cur.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(contents) = fs::read_to_string(&candidate) {
                if contents.contains("[workspace]") {
                    return Some(cur);
                }
            }
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

fn discover_targets(workspace_root: &Path, repo_root: &Path) -> Result<Vec<Target>, String> {
    let members = parse_workspace_members(workspace_root)?;
    let mut targets: Vec<Target> = Vec::new();

    for member in members {
        let crate_dir = workspace_root.join(&member);
        if !crate_dir.exists() {
            continue;
        }
        let crate_name = member.split('/').last().unwrap_or(&member).to_string();

        // Examples
        let examples_dir = crate_dir.join("examples");
        if examples_dir.exists() {
            for entry in read_rs_files(&examples_dir)? {
                let name = entry.file_stem().unwrap_or_default().to_string_lossy().to_string();
                targets.push(Target {
                    name,
                    crate_name: Some(crate_name.clone()),
                    kind: TargetKind::Example,
                    path: Some(entry),
                });
            }
        }

        // Bins from src/bin
        let bins_dir = crate_dir.join("src/bin");
        if bins_dir.exists() {
            for entry in read_rs_files(&bins_dir)? {
                let name = entry.file_stem().unwrap_or_default().to_string_lossy().to_string();
                targets.push(Target {
                    name,
                    crate_name: Some(crate_name.clone()),
                    kind: TargetKind::Bin,
                    path: Some(entry),
                });
            }
        }

        // Default bin from src/main.rs
        if crate_dir.join("src/main.rs").exists() {
            targets.push(Target {
                name: crate_name.clone(),
                crate_name: Some(crate_name.clone()),
                kind: TargetKind::Bin,
                path: Some(crate_dir.join("src/main.rs")),
            });
        }
    }

    // Scripts from repo_root/scripts
    let scripts_dir = repo_root.join("scripts");
    if scripts_dir.exists() {
        for entry in fs::read_dir(&scripts_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                continue;
            }
            let file_name = match path.file_name() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            };
            let name = file_name.strip_suffix(".sh").unwrap_or(&file_name).to_string();
            targets.push(Target {
                name,
                crate_name: None,
                kind: TargetKind::Script,
                path: Some(path),
            });
        }
    }

    Ok(targets)
}

fn read_rs_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "rs" {
                    files.push(path);
                }
            }
        }
    }
    Ok(files)
}

fn parse_workspace_members(workspace_root: &Path) -> Result<Vec<String>, String> {
    let cargo_toml = workspace_root.join("Cargo.toml");
    let contents = fs::read_to_string(&cargo_toml).map_err(|e| e.to_string())?;

    let mut members: Vec<String> = Vec::new();
    let mut in_members = false;

    for line in contents.lines() {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("members") && trimmed.contains('[') {
            in_members = true;
        }
        if in_members {
            let mut chars = trimmed.chars().peekable();
            while let Some(ch) = chars.next() {
                if ch == '"' {
                    let mut value = String::new();
                    while let Some(c) = chars.next() {
                        if c == '"' {
                            break;
                        }
                        value.push(c);
                    }
                    if !value.is_empty() {
                        members.push(value);
                    }
                }
            }
            if trimmed.contains(']') {
                in_members = false;
            }
        }
    }

    if members.is_empty() {
        return Err("No workspace members found".to_string());
    }

    Ok(members)
}
