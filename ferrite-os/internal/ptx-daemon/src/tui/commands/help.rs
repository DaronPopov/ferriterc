use crate::events::{LogCategory, LogEntry};
use crate::tui::state::TuiState;

pub(super) fn cmd_help(state: &mut TuiState) {
    // Navigation — always useful, shown first
    state.push_log(LogEntry::new(LogCategory::Jit, "── navigation ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ctrl+o           toggle shell/files mode",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /sysmon [on|off]  visual telemetry dashboard",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /detail          toggle tensor/pipeline detail",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /density         set ui density: auto/compact/balanced/comfortable",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /fxscript        load scripted tachyonfx effects (status/off/reset/load)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /plot3d         3D render window control (status/open/close/scene/tensor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  tab              toggle log detail/summary view",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ↑↓               scroll sysmon sections (small view) or log/history",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pgup/pgdn        jump section viewport when sysmon overflows",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── processes ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ps               list running processes",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  start <app>      launch managed app",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  kill <id|name>   stop a process",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  uptime           daemon uptime + request counts",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── memory ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pool             vram pool status",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pool check       deep pool validation",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  defrag           defragment vram",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── gpu ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  stats            runtime stats",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  hwpoll           hardware polling status (nvml/cupti)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  streams          stream pool info",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  snapshot         system state snapshot",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  health           health check",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── demos ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  bench            gpu compute benchmark",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demos            list available demo programs",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo <name>      run a demo program",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo stop        halt running stress demo",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo gpu-logwatch GPU log stream monitor (continuous)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo dataflow-proof prove TPU-like memory determinism (continuous)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  demo inspect <n> open demo source in editor",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── workspace ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /files           open file explorer/editor",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /open <path>     open file directly",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /save            save current file",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  clear            clear log",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quit             shutdown daemon",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── filesystem ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  mkdir <path>     create directory",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  touch <path>     create empty file",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  mv <src> <dst>   move/rename file or directory",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  cp <src> <dst>   copy file",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  rm <path>        remove file/dir (requires confirm)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ls [path]        list directory contents",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  cd [path]        change working directory",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  pwd              show working directory",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── run ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  run              compile+execute open file (F5 in editor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /stop            stop running program (F6 in editor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  rerun            re-run last file",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  args <...>       set run arguments",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  profile [d|r]    set debug/release profile",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  timeout [secs]   set/show execution timeout (default 60s)",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── inspect / profiling ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  ptx              inspect last compiled program (opens buffer in editor)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  perf             kernel profiling history + statistics",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  jit-clear        clear JIT compilation cache",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  jit-cache        show JIT cache stats (memory + disk)",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── vram quota ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota            show per-owner VRAM usage",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota set <id> <mb>  set soft limit (MB) for owner",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota clear <id>     remove soft limit for owner",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  quota label <id> <n> name an owner",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── editor (vim-like) ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :help            full editor command reference (in editor mode)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :keymap          keymap/remapping help",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :map             list current keybindings",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :map <key> <act> remap key to action (e.g. :map <C-p> SearchForward)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :actions         list all remappable actions",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :marks           list marks  |  :registers  list registers",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :grep <pat>      workspace search  |  :s/pat/rep/g  substitute",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  :buffers         list open buffers  |  :bn/:bp  switch buffer",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── agent ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  audit            show last 20 agent actions",
    ));

    state.push_log(LogEntry::new(LogCategory::Jit, "── scheduler / control plane ──"));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler        open scheduler dashboard (ctrl+o to exit)",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler pause  pause the scheduler queue",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler resume resume the scheduler queue",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler stats  scheduler statistics",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  scheduler policies  list active policy rules",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  /policy          show policy status and recent decisions",
    ));
    state.push_log(LogEntry::new(
        LogCategory::Sys,
        "  audit-query [--tenant ID] [--last N]  query audit log",
    ));
}
