# Safe Edit Rules

## Rule 1: Keep Changes Layer-Scoped

Do not edit runtime, installer, and integration layers together unless required.

## Rule 2: Preserve Compatibility Contracts

When changing compatibility behavior:

- update `compat.toml`
- update resolver behavior
- update install docs

All three must remain consistent.

## Rule 3: Prefer Additive Changes

- Add new feature keys instead of removing existing ones unless explicitly requested.
- Keep defaults stable unless there is a clear migration path.

## Rule 4: Validate Script Syntax

After shell changes, run:

```bash
bash -n install.sh
bash -n ferrite-run
bash -n scripts/resolve_cuda_compat.sh
bash -n scripts/install/lib/*.sh
```

## Rule 5: Include Minimal Verification Notes

Any change should include the command(s) used to verify behavior.

## Rule 6: Avoid Implicit Behavior Changes

If behavior changes, document it in:

- `INSTALL.md`
- install section in `README.md`

