# WSL LeWM Training Boundary

Status: boundary policy draft
Scope: LeWM/world-model training execution under WSL/Linux

---

## 0. 결론

LeWM 학습은 Windows 루트에서 직접 실행하지 않는다. 학습 실행부는 `wsl/lewm/` 아래에 두고, 실제 실행은 WSL/Linux 환경에서 수행한다.

```yaml
decision:
  lewm_training_runtime: WSL/Linux only
  windows_repo_root: docs, schemas, source audit, qwen baseline, router/eval orchestration
```

---

## 1. 이유

```yaml
why_wsl:
  - LeWM 공식 stack은 Linux 환경에서 더 안정적임
  - dataset/cache/checkpoint I/O가 많아 Windows path issue를 줄여야 함
  - PyTorch/Lightning/Hydra runtime 로그와 checkpoint를 Linux filesystem에 두는 편이 안전함
  - Codex 앱 경로 참조 때문에 repo folder name은 유지하되 training execution만 분리함
```

---

## 2. Repo Boundary

```yaml
windows_or_codex_root:
  path: <repo-root-on-windows>
  owns:
    - README.md
    - docs/
    - schemas/
    - configs/data
    - configs/qwen
    - configs/router
    - configs/eval
    - scripts for audit, qwen baseline, probe/eval orchestration

wsl_lewm_zone:
  repo_path: wsl/lewm
  runtime_path_recommended: <wsl-home>/circuit-lewm-routing
  current_m5_runtime: external_official_lewm_checkout
  owns:
    - LeWM training wrappers
    - Linux-specific Hydra/config overrides
    - latent export utilities
```

`wsl/lewm/` 아래 versioned script/config는 commit할 수 있지만, runtime output은 commit하지 않는다.

---

## 3. Local-Only Outputs

```yaml
ignored_outputs:
  - wsl/lewm/.venv/
  - wsl/lewm/data/
  - wsl/lewm/runs/
  - wsl/lewm/outputs/
  - wsl/lewm/checkpoints/
  - wsl/lewm/wandb/
```

WSL 학습 결과 중 commit 가능한 것은 작은 summary뿐이다.

```yaml
commit_allowed_after_training:
  - results/lewm_s/pretrain_log_summary.json
  - docs/20_results/005_lewm_s_pretraining_ko.md
  - configs used for the run
  - source/split manifest

commit_forbidden_after_training:
  - full checkpoint
  - tensor dump
  - raw image cache
  - full wandb run directory
```

---

## 4. 실행 원칙

```text
Windows/Codex:
  source audit
  manifest generation
  Qwen baseline
  result review

WSL/Linux:
  LeWM dataset materialization
  LeWM-S / LeWM-M training
  checkpoint save
  latent export
```

Qwen baseline과 LeWM training은 같은 process에서 섞지 않는다. memory/latency 측정도 각각 clean process에서 수행한다.
