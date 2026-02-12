# Prompting Contract (LLM -> Repo)

## Input Expectations

A good task prompt should specify:

1. Desired behavior change.
2. Scope limits (files/layers to avoid changing).
3. Whether backward compatibility is required.
4. Validation expectations (syntax check, build check, runtime check).

## Expected LLM Output for Code Tasks

1. Short statement of implemented change.
2. Exact files changed.
3. Verification performed.
4. Remaining risks/assumptions.

## Required Operational Behaviors

- Use repository-local paths in responses.
- Keep commands copy-paste ready.
- Avoid destructive git operations unless explicitly requested.
- Prefer factual statements over claims.

## Example Task Template

```text
Update installer compatibility mapping for CUDA 12.9.
Only touch install/compat docs and scripts.
Keep existing defaults unchanged.
Scope: full mode (not --core-only).
Run shell syntax checks after edits.
```

## Example Completion Template

```text
Changed:
- compat.toml
- scripts/resolve_cuda_compat.sh
- INSTALL.md

Validation:
- bash -n scripts/resolve_cuda_compat.sh
- scripts/resolve_cuda_compat.sh --format env

Risk:
- Not tested on physical CUDA 12.9 host in this environment.
```

