# WSL LeWM Training Zone

Status: Linux-only training scaffold

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

Use a Linux-native runtime checkout such as `/home/user/circuit-lewm-routing` when full training I/O becomes heavy.
