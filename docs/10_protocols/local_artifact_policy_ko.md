# Local Artifact Policy

Status: policy draft
Scope: local models, datasets, checkpoints, caches, logs

---

## 0. 원칙

이 레포는 public repo로 운영할 수 있어야 한다. 따라서 대용량 모델, 원본 데이터셋, checkpoint, cache, log는 git에 올리지 않는다.

```text
Docs, configs, schemas, small examples: commit allowed
Models, datasets, checkpoints, raw logs: local only
```

---

## 1. Git에 올리지 않는 것

```yaml
local_only:
  - .local/
  - models/
  - checkpoints/
  - data/raw/
  - data/cache/
  - data/downloads/
  - artifacts/
  - logs/
  - runs/
  - outputs/
  - wandb/
  - wsl/lewm/data/
  - wsl/lewm/runs/
  - wsl/lewm/checkpoints/
  - wsl/lewm/outputs/
  - large binary model files
  - raw downloaded datasets
```

확장자 기준으로도 다음 파일은 commit하지 않는다.

```yaml
ignored_extensions:
  - "*.ckpt"
  - "*.pt"
  - "*.pth"
  - "*.safetensors"
  - "*.gguf"
  - "*.bin"
  - "*.onnx"
  - "*.parquet"
  - "*.arrow"
  - "*.lance"
  - "*.h5"
  - "*.hdf5"
```

---

## 2. Git에 올릴 수 있는 것

```yaml
commit_allowed:
  - README.md
  - docs/**/*.md
  - configs/**/*.yaml
  - schemas/**/*.json
  - schemas/**/*.yaml
  - scripts/**/*.py
  - small hand-written examples
  - source manifests without embedded raw data
  - WSL/Linux wrapper scripts without embedded local secrets
```

Dataset provenance는 기록하되, 원본 데이터 자체는 올리지 않는다.

---

## 3. Source Manifest 원칙

Dataset audit가 끝나면 다음 정보를 작은 manifest로 남긴다.

```yaml
source_manifest_fields:
  - dataset_id
  - source_url
  - license
  - split_used
  - usable_rows
  - excluded_reason_summary
  - local_cache_path_optional
  - audit_date
```

`local_cache_path_optional`은 로컬 경로 힌트일 뿐이며, public repo에서 재현 가능한 source id와 split 정보를 우선한다.

---

## 4. 결과 보존 원칙

결과는 작은 요약 JSON과 markdown report만 commit한다.

```yaml
commit_result:
  - aggregate score table
  - command/config summary
  - source/split provenance
  - failure mode summary

do_not_commit_result:
  - full raw model output dump if large
  - image cache
  - checkpoint
  - tensor dump
  - private/local absolute cache bundle
```
