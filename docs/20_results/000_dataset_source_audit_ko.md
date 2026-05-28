# M1 Dataset Source Audit

Status: M1 closed with caveats
Date: 2026-05-29 KST
Manifest: `data/circuit_sources/source_manifest.json`
Script: `scripts/audit_sources.py`

---

## 0. 결론

M1의 목적은 회로 도메인 Perception LeWM 학습과 Qwen/Router 비교 평가에 사용할 데이터셋의 출처, 라이선스, 크기, 다운로드 전략, split 위험을 정리하는 것이다.

이번 M1에서는 전체 원본 데이터를 로컬에 다운로드하지 않았다. HF Dataset Viewer가 일부 row page에서 500 오류를 낼 수 있으므로, M1은 Viewer metadata, parquet size, first-row field preview, Hub file inventory만 기록한다.

가장 중요한 결론은 다음과 같다.

```yaml
primary_sources:
  structure_world_model_pretraining:
    - bshada/open-schematics
  structure_text_prior:
    - microsoft/SchGen_dataset
  vqa_evaluation:
    - ayoubkirouane/CircuitVQA
  perception_probe:
    - lowercaseonly/cghd

download_strategy:
  full_open_schematics_download_in_m1: avoid
  m2_required_action: build usable image+schematic pair curriculum, target at least 5k train pairs if available
  commit_raw_payloads: false
```

---

## 1. Local Download Footprint

확인된 압축 parquet 기준 용량은 다음과 같다.

```yaml
bshada/open-schematics:
  rows: 84470
  parquet: 6.21 GiB
  decoded_memory_estimate: 14.03 GiB
  estimated_5k_parquet_equivalent: 376.32 MiB
  recommendation: avoid full snapshot in M1; build 5k usable-pair cache in M2

microsoft/SchGen_dataset:
  rows: 8420
  parquet: 128.28 MiB
  decoded_memory_estimate: 128.25 MiB
  recommendation: acceptable if needed

ayoubkirouane/CircuitVQA:
  rows: 10470
  parquet: 363.81 MiB
  decoded_memory_estimate: 373.95 MiB
  recommendation: metadata/eval cache only until license is clarified

lowercaseonly/cghd:
  rows: not exposed by Dataset Viewer
  file_inventory: 8207 siblings
  recommendation: partial/pattern download only after image/XML pairing plan
```

`open-schematics` 전체를 `datasets.load_dataset()` 방식으로 받으면 parquet 6.21GiB 외에 Arrow/cache/materialized image 부담이 추가될 수 있다. 그래서 디스크 여유는 최소 15~25GiB 정도를 잡는 편이 안전하다.

5k 학습 curriculum을 목표로 하면 전체 snapshot보다 훨씬 작게 시작할 수 있다. 다만 parquet shard 단위 다운로드와 row-level subset cache의 실제 크기는 M2에서 측정해야 한다.

---

## 2. Dataset Decisions

### 2.1 bshada/open-schematics

```yaml
source_url: https://huggingface.co/datasets/bshada/open-schematics
decision: selected_for_structure_pretraining
license: cc-by-4.0
rows: 84470
fields:
  - schematic
  - image
  - components_used
  - json
  - yaml
  - name
  - description
  - type
role:
  - image-to-structure alignment
  - LeWM structure/world-model pretraining
```

주의:

```yaml
caveats:
  - full local download is large enough to defer until M2/M5 needs it
  - some rows can have missing image payloads
  - M2 must build a usable image+schematic pair manifest, not just a row-reference manifest
  - project/name near-duplicate leakage must be checked before paper-ready training
```

### 2.2 microsoft/SchGen_dataset

```yaml
source_url: https://huggingface.co/datasets/microsoft/SchGen_dataset
decision: selected_for_structure_text_prior
license: mit
rows: 8420
parquet: 128.28 MiB
role:
  - circuit structure/text prior
  - KiCad generation prior
  - teacher/curriculum support
```

주의:

```yaml
caveats:
  - not an image perception dataset
  - do not mix into VQA evaluation evidence
```

### 2.3 ayoubkirouane/CircuitVQA

```yaml
source_url: https://huggingface.co/datasets/ayoubkirouane/CircuitVQA
decision: selected_for_vqa_evaluation_with_license_caveat
license: not_declared_on_hf
rows:
  train: 8376
  test: 2094
  total: 10470
parquet: 363.81 MiB
role:
  - Qwen baseline comparison
  - end-to-end QA evaluation
```

주의:

```yaml
caveats:
  - license is not declared on HF
  - prompt/answer payload should not be committed
  - answer normalization policy is required before official scoring
```

### 2.4 lowercaseonly/cghd

```yaml
source_url: https://huggingface.co/datasets/lowercaseonly/cghd
decision: selected_for_perception_probe
license: cc-by-3.0
file_inventory:
  total_siblings: 8207
  image_files: 4537
  xml_annotation_files: 3293
  json_annotation_files: 358
role:
  - handwritten symbol grounding
  - object detection / segmentation probe
  - LeWM latent evidence retention probe
```

주의:

```yaml
caveats:
  - Dataset Viewer does not expose normal row/size table
  - full file-tree byte size is not cheaply available from Viewer metadata
  - M2 must create explicit image/XML pairing manifest
```

---

## 3. Secondary / Excluded Sources

```yaml
MirandaAbhilash/circuitvqa-dataset:
  decision: optional_supplementary_image_candidate
  license: not_declared_on_hf
  parquet: 83.66 MiB
  caveat: label semantics are insufficient for primary VQA supervision

Ailiance-fr/kicad9plus-copyleft:
  decision: exclude_from_initial_training_due_to_copyleft_scope
  license: gpl-3.0
  parquet: 1.60 MiB
  caveat: copyleft obligations need explicit acceptance

hanky2397/schematic_images:
  decision: exclude_initially_due_to_gated_access_and_missing_license
  gated: auto
  rough_zip_inventory_from_hub_tree_probe: about 1.62 GiB
  caveat: do not use until access terms and license are reviewed
```

---

## 4. M1 Pass Check

```yaml
pass_if:
  selected_source_datasets_have_provenance_notes: true
  license_or_license_caveat_recorded: true
  size_and_download_strategy_recorded: true
  split_leakage_risk_documented: true
  ambiguous_gated_copyleft_sources_excluded_or_optional: true

status: closed_with_caveats
next_milestone: M2 usable circuit curriculum and split
```

M2에서 반드시 고칠 점:

```yaml
m2_requirements:
  - do not treat row-reference count as usable training pair count
  - verify actual image+schematic payload before adding an open-schematics row to train/holdout/test
  - target at least 5k usable train pairs if source availability allows it
  - keep raw images, prompts, answers, and signed URLs out of committed artifacts
```

---

## 5. Reproducibility

```powershell
python .\scripts\audit_sources.py
```

Output:

```text
data/circuit_sources/source_manifest.json
```
