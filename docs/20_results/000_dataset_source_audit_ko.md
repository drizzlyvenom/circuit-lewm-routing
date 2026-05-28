# M1 Dataset Source Audit

Status: M1 closed with caveats
Date: 2026-05-29 KST
Manifest: `data/circuit_sources/source_manifest.json`
Audit script: `scripts/audit_sources.py`

---

## 0. 결론

M1의 목적은 회로 도메인 LeWM 실험에 사용할 dataset source를 역할별로 나누고, 출처/라이선스/split/누수 위험을 명확히 기록하는 것이다.

이번 audit 결과, 초기 실험 source는 다음처럼 확정한다.

```yaml
selected_sources:
  structure_world_model_pretraining:
    - bshada/open-schematics

  structure_text_prior:
    - microsoft/SchGen_dataset

  vqa_evaluation:
    - ayoubkirouane/CircuitVQA
    - note: license not declared on HF; use with license caveat

  perception_probe:
    - lowercaseonly/cghd

optional_or_excluded:
  MirandaAbhilash/circuitvqa-dataset: optional supplementary image candidate
  Ailiance-fr/kicad9plus-copyleft: excluded initially due to GPL-3.0/copyleft scope
  hanky2397/schematic_images: excluded initially due to gated access and missing license
```

M1은 다음 단계로 넘어갈 수 있다. 다만 `CircuitVQA`는 HF card에 명시 라이선스가 없어 paper-ready evaluation 전에 라이선스 확인이 필요하다.

---

## 1. Primary Sources

### 1.1 bshada/open-schematics

```yaml
source_url: https://huggingface.co/datasets/bshada/open-schematics
decision: selected_for_structure_pretraining
license: cc-by-4.0
rows:
  train: 84470
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
  - structure/world-model pretraining
  - image-to-structure alignment
  - circuit component metadata alignment
```

판단:

```text
M1 기준 가장 중요한 source다.
이미지, KiCad schematic, component list, JSON/YAML structure가 함께 있으므로
Perception LeWM의 image-to-structure alignment에 직접 사용할 수 있다.
```

주의:

```yaml
caveats:
  - final split은 random row split보다 project/name grouped split을 우선한다.
  - source project license compliance는 CC-BY-4.0 외에도 upstream project license를 함께 확인한다.
  - 일부 project metadata 품질이 다를 수 있으므로 malformed/empty schematic filtering을 유지한다.
```

### 1.2 microsoft/SchGen_dataset

```yaml
source_url: https://huggingface.co/datasets/microsoft/SchGen_dataset
decision: selected_for_structure_text_prior
license: mit
rows:
  train: 8420
fields:
  - messages
  - meta
role:
  - circuit structure/text prior
  - KiCad generation prior
  - curriculum/teacher support
```

판단:

```text
이미지 perception source가 아니라 circuit structure/code/text prior다.
LeWM image encoder를 직접 학습시키는 용도보다,
structure encoder, teacher prompt, KiCad schema normalization에 적합하다.
```

주의:

```yaml
caveats:
  - VQA evaluation split과 섞지 않는다.
  - meta의 로컬 path는 upstream generation context이며 local path로 재현 가능한 경로가 아니다.
```

### 1.3 ayoubkirouane/CircuitVQA

```yaml
source_url: https://huggingface.co/datasets/ayoubkirouane/CircuitVQA
decision: selected_for_vqa_evaluation_with_license_caveat
license: not_declared_on_hf
rows:
  train: 8376
  test: 2094
  total: 10470
fields:
  - texts
  - images
role:
  - circuit image question answering
  - Qwen3 baseline comparison
  - end-to-end LeWM system evaluation
```

판단:

```text
Qwen3 단일 backbone과 LeWM system을 직접 비교하기 좋은 VQA source다.
각 row에 image list와 user/assistant text list가 있어 QA extraction/evaluation으로 변환 가능하다.
```

주의:

```yaml
caveats:
  - HF card에 license가 명시되지 않았다.
  - README content is empty로 표시되므로 provenance와 original source 확인이 필요하다.
  - train/test 사이 image 중복 또는 같은 base image의 변형이 있는지 dedup audit을 해야 한다.
  - paper-ready result에는 license 확인 후 포함한다.
```

### 1.4 lowercaseonly/cghd

```yaml
source_url: https://huggingface.co/datasets/lowercaseonly/cghd
decision: selected_for_perception_probe
license: cc-by-3.0
hf_tags:
  - object-detection
  - image-segmentation
file_inventory:
  total_files: 8207
  image_files: 4537
  xml_annotation_files: 3293
  json_annotation_files: 358
role:
  - handwritten circuit symbol grounding
  - object detection / segmentation probe
  - LeWM latent evidence retention probe
```

판단:

```text
VQA source가 아니라 perception probe source다.
LeWM latent가 symbol, text, stroke, component evidence를 보존하는지 확인하는 데 쓴다.
```

주의:

```yaml
caveats:
  - Dataset Viewer size/parquet는 정상 row table로 제공되지 않았다.
  - file-tree dataset이므로 image/annotation pairing manifest를 별도로 만들어야 한다.
  - train/holdout/test split은 drafter/circuit id 기준 grouped split을 새로 정의한다.
```

---

## 2. Secondary / Excluded Sources

### 2.1 MirandaAbhilash/circuitvqa-dataset

```yaml
source_url: https://huggingface.co/datasets/MirandaAbhilash/circuitvqa-dataset
decision: optional_supplementary_image_candidate
license: not_declared_on_hf
rows:
  train: 5963
  validation: 846
  test: 1706
  total: 8515
fields:
  - image
  - label
```

판단:

```text
이미지 보조 후보로는 쓸 수 있으나, 현재 preview만으로는 VQA supervision이 충분히 명확하지 않다.
license와 label semantics를 확인하기 전에는 primary source로 쓰지 않는다.
```

### 2.2 Ailiance-fr/kicad9plus-copyleft

```yaml
source_url: https://huggingface.co/datasets/Ailiance-fr/kicad9plus-copyleft
decision: exclude_from_initial_training_due_to_copyleft_scope
license: gpl-3.0
rows:
  train: 209
fields:
  - messages
  - metadata
```

판단:

```text
KiCad text corpus로는 유용하지만 GPL-3.0/copyleft obligation이 강하다.
초기 public experiment에는 넣지 않고, 필요 시 별도 license decision 이후 사용한다.
```

### 2.3 hanky2397/schematic_images

```yaml
source_url: https://huggingface.co/datasets/hanky2397/schematic_images
decision: exclude_initially_due_to_gated_access_and_missing_license
license: not_declared_on_hf
gated: auto
files:
  - images.zip
  - components.zip
  - pkl.zip
  - sp.zip
```

판단:

```text
schematic image to HSPICE netlist task에는 흥미롭지만,
접근 조건 동의가 필요하고 license metadata가 비어 있어 M1 초기 source에서는 제외한다.
```

---

## 3. Split / Leakage Policy

```yaml
open_schematics:
  preferred_split_unit:
    - project name
    - repository/name field
  avoid:
    - random row split before near-duplicate audit

SchGen_dataset:
  preferred_split_unit:
    - module
    - schematic_path basename
  avoid:
    - mixing structure text prior into VQA test evidence

CircuitVQA:
  preferred_split_unit:
    - provided train/test split first
  required_before_paper:
    - image hash dedup
    - repeated question/image family check
    - answer normalization policy

CGHD:
  preferred_split_unit:
    - drafter id
    - circuit id
  required:
    - image/annotation pairing manifest
    - grouped holdout split
```

---

## 4. M1 Pass Check

```yaml
pass_if:
  all_selected_source_datasets_have_provenance_notes: true
  license_or_license_caveat_recorded: true
  usable_image_or_structure_counts_recorded: true
  split_leakage_risk_documented: true
  ambiguous_gated_copyleft_sources_excluded_or_optional: true
```

M1 status:

```yaml
status: closed_with_caveats
next_milestone: M2 Circuit sample schema and splits
blocking_before_paper_ready:
  - CircuitVQA license clarification
  - open-schematics upstream license attribution policy
  - CGHD image/annotation pairing manifest
  - split leakage audit
```

---

## 5. Reproducibility

Audit command:

```powershell
python .\scripts\audit_sources.py
```

Output:

```text
data/circuit_sources/source_manifest.json
```

이 manifest에는 HF dataset URL, Hub API URL, Dataset Viewer split/size/parquet 상태, field preview, file inventory, M1 decision이 포함된다. Signed image asset URL이나 token은 저장하지 않는다.
