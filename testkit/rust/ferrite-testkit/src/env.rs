use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

pub fn resolve_repo_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }

    let mut cur = std::env::current_dir()?;
    loop {
        if cur.join("ferrite-os").join("Cargo.toml").exists() && cur.join(".git").exists() {
            return Ok(cur);
        }
        if !cur.pop() {
            break;
        }
    }

    Err(anyhow!("unable to locate repo root (expected ferrite-os/Cargo.toml + .git)"))
}

pub fn resolve_daemon_binary(repo_root: &Path) -> Option<PathBuf> {
    let release = repo_root
        .join("ferrite-os")
        .join("target")
        .join("release")
        .join("ferrite-daemon");
    if release.exists() {
        return Some(release);
    }

    let debug = repo_root
        .join("ferrite-os")
        .join("target")
        .join("debug")
        .join("ferrite-daemon");
    if debug.exists() {
        return Some(debug);
    }

    None
}

pub fn default_daemon_config(repo_root: &Path) -> PathBuf {
    repo_root
        .join("ferrite-os")
        .join("internal")
        .join("ptx-daemon")
        .join("dev-config.toml")
}
