# WSL LeWM Training Zone

Status: Linux-only training wrappers active

This folder owns LeWM/world-model training wrappers and Linux-specific runtime configs.

Recommended execution boundary:

```text
Windows/Codex repo root:
  docs, schemas, source audit, Qwen baseline, router/eval orchestration

WSL/Linux:
  LeWM dataset materialization, training, checkpoints, latent export
```

Commit allowed:

```text
wsl/lewm/README.md
wsl/lewm/configs/*
wsl/lewm/scripts/*
wsl/lewm/manifests/*
```

Commit forbidden:

```text
wsl/lewm/.venv/
wsl/lewm/data/
wsl/lewm/runs/
wsl/lewm/outputs/
wsl/lewm/checkpoints/
wsl/lewm/wandb/
```

M5 uses the external official LeWM checkout as the runtime environment and keeps checkpoints/step logs out of this repository. Commit only wrappers, configs, manifests, and compact result summaries.
