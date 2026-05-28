# Circuit LeWM + LoRA Routing 검증 마일스톤

Status: milestone draft
Scope: circuit-domain Perception LeWM / LLM router / JEPA router validation
Target hardware: RTX 3090, usable VRAM 24GB, usable RAM 24~26GB

---

## 0. 핵심 질문

```text
Perception LeWM이 회로 구조 evidence를 작은 latent에 충분히 보존한다면,
작은 router / answer head / certified adapter만으로
단일 Qwen3-VL backbone 대비 더 낮은 resident/peak VRAM에서
허용 가능한 회로 QA 및 structure extraction 성능을 낼 수 있는가?
```

핵심 원칙:

```yaml
main_claim:
  - compact circuit perception/world-model backbone
  - Qwen3 monolithic VLM 대비 resource/accuracy tradeoff
  - LeWM latent evidence retention
  - LLM router vs JEPA router 비교

not_main_claim:
  - LoRA alone reduces Qwen3 resident VRAM
  - Foveation novelty
  - Qwen-trained LoRA transfer to LeWM
  - proxy certification
```

---

## 1. 전체 마일스톤

```text
M0. Repo scaffold and boundary
M1. Dataset source audit
M2. Circuit sample schema and splits
M3. Qwen baseline measurement
M4. LeWM data pipeline
M5. LeWM-S structure pretraining
M6. Frozen latent probe
M7. Router contract and oracle baseline
M8. LLM router baseline
M9. JEPA latent router baseline
M10. Answer head / adapter head evaluation
M11. Actual-only certification registry
M12. End-to-end system comparison
M13. LeWM-M scaling decision
M14. Paper-ready result table
```

핵심 gate:

```yaml
Gate_D:
  name: dataset_audit
  asks: "source/license/split leakage가 안전한가?"

Gate_P:
  name: perception_evidence_retention
  asks: "LeWM latent가 회로 evidence를 보존하는가?"

Gate_R:
  name: router_selection
  asks: "router가 taxonomy/adapter를 oracle에 가깝게 고르는가?"

Gate_A:
  name: answer_task
  asks: "LeWM system이 Qwen baseline 대비 허용 가능한 score를 내는가?"

Gate_C:
  name: cost_tradeoff
  asks: "resource/latency/parameter에서 monolithic Qwen보다 유리한가?"
```

---

## M0. Repo Scaffold and Boundary

### 목표

새 레포를 `Circuit LeWM + Routing` 중심으로 시작한다.

### 작업

```yaml
tasks:
  - create README with circuit LeWM thesis
  - add framework/proposal doc
  - add validation milestone doc
  - add local artifact policy
  - add WSL/Linux LeWM training boundary policy
  - add result brief policy
  - import or copy minimal AdapterCard / RouteTrace schema from taxonomy-lora repo
  - create repo folder scaffold
```

### 권장 폴더

```text
README.md

docs/
  00_overview/
    framework_ko.md
    repo_boundary_ko.md
  10_protocols/
    circuit_lewm_validation_milestones_ko.md
    qwen_vs_lewm_comparison_protocol_ko.md
    circuit_lewm_training_reference_ko.md
    actual_certification_protocol_ko.md
    local_artifact_policy_ko.md
    wsl_lewm_training_boundary_ko.md
  20_results/
    README.md
  30_paper_notes/
    paper_outline_ko.md
    claim_boundary_ko.md

schemas/
  circuit_sample.example.json
  lewm_pretrain_trace.example.yaml
  lewm_probe_result.example.yaml
  router_eval_result.example.yaml
  system_comparison_result.example.yaml
  adapter_card_v3.example.yaml

configs/
  data/
  lewm/
  qwen/
  router/
  eval/

src/
  circuit_lewm/
    data/
    models/
    objectives/
    probes/
    routers/
    eval/
    registry/

scripts/
  audit_sources.py
  prepare_circuit_samples.py
  run_qwen_baseline.py
  run_lewm_probe.py
  train_router_llm.py
  train_router_jepa.py
  run_system_comparison.py

wsl/
  lewm/
    README.md
    configs/
    scripts/
      train_lewm.py
      export_latents.py
    manifests/
```

역할 분리:

```yaml
windows_or_codex_root:
  owns:
    - docs
    - schemas
    - dataset source audit
    - Qwen baseline scripts
    - router/eval orchestration
    - result summaries

wsl_linux_lewm_zone:
  path: wsl/lewm
  owns:
    - LeWM training entrypoints
    - Linux/Hydra/runtime configs
    - latent export utilities
  local_only_outputs:
    - wsl/lewm/data/
    - wsl/lewm/runs/
    - wsl/lewm/checkpoints/
```

### 통과 조건

```yaml
pass_if:
  - README says LeWM experiment is separate from taxonomy-lora protocol repo
  - local artifact policy exists
  - WSL/Linux LeWM training boundary is documented
  - no raw datasets/checkpoints are committed
  - AdapterCard/certification schema imported without proxy fields
  - repo folder scaffold exists
```

---

## M1. Dataset Source Audit

### 목표

open-schematics / SchGen / CircuitVQA / CGHD를 역할별로 검증한다.

### 데이터셋 역할

```yaml
open_schematics:
  role:
    - structure/world-model pretraining
    - image-to-structure alignment

SchGen:
  role:
    - circuit structure/text/code prior
    - curriculum/teacher support

CircuitVQA:
  role:
    - end-to-end QA evaluation
    - Qwen baseline comparison

CGHD:
  role:
    - symbol grounding / perception probe
    - handwritten robustness probe
```

### 작업

```yaml
tasks:
  - verify license
  - count usable rows
  - inspect image/structure fields
  - check malformed samples
  - check duplicate or near-duplicate leakage
  - define split unit
  - record source provenance
```

### 산출물

```text
data/circuit_sources/source_manifest.json
docs/20_results/000_dataset_source_audit_ko.md
```

### 통과 조건

```yaml
pass_if:
  - all selected source datasets have license/provenance notes
  - usable image/structure pairs are counted
  - split leakage risk is documented
  - ambiguous/gated/copyleft sources are marked optional or excluded
```

Current closure:

```yaml
status: closed_with_caveats
manifest: data/circuit_sources/source_manifest.json
result_brief: docs/20_results/000_dataset_source_audit_ko.md
main_caveats:
  - open-schematics full parquet footprint is about 6.21 GiB, so M1 avoids full local snapshot
  - row count is not the same as usable image+schematic pair count
  - CircuitVQA license is not declared on HF
  - CGHD is file-tree based and requires explicit image/XML pairing
next_milestone_requirement:
  - M2 must build a usable training curriculum, targeting at least 5k verified open-schematics image+schematic train pairs if available
```

---

## M2. Circuit Sample Schema and Splits

### 목표

모든 후속 실험이 같은 sample schema를 쓰게 하고, 실제 학습 가능한 usable circuit curriculum을 만든다.

### Schema

```yaml
CircuitSample:
  sample_id: string
  source_dataset: string
  split: train | holdout | test

  image:
    image_path_or_ref: string
    width: int | null
    height: int | null

  query:
    prompt: string | null
    expected_answers: list[str] | null
    answer_type: string | null

  structure:
    structure_path_or_ref: string | null
    component_list: list | null
    netlist_or_edges: list | null
    metadata_text: string | null

  supervision:
    taxonomy: dict | null
    hard_negatives: list[str]
    roi_or_tile_boxes: list | null
```

### 산출물

```text
data/circuit_curricula/train.jsonl
data/circuit_curricula/holdout.jsonl
data/circuit_curricula/test.jsonl
data/circuit_curricula/usable_pair_summary.json
docs/20_results/001_circuit_sample_schema_ko.md
```

### 통과 조건

```yaml
pass_if:
  - train/holdout/test split exists
  - open-schematics rows are verified as actual image+schematic pairs before entering the training curriculum
  - train split targets at least 5k usable image+schematic pairs if source availability allows it
  - source leakage policy is documented
  - answer_type is present for QA samples
  - structure fields exist for structure pretraining samples
```

---

## M3. Qwen Baseline Measurement

### 목표

강한 monolithic baseline과 작은/quantized baseline을 확정한다.

### 비교군

```yaml
baselines:
  qwen3_vl_single:
    role: strong_monolithic_baseline

  smaller_or_quantized_qwen:
    role: practical_low_vram_monolithic_baseline
```

### 측정

```yaml
metrics:
  - circuit_vqa_score
  - structured_extraction_score_if_available
  - resident_vram_mb
  - peak_vram_mb
  - latency_ms
  - total_parameters
```

### 산출물

```text
results/qwen/qwen3_single_backbone.json
results/qwen/qwen_small_or_quantized.json
docs/20_results/002_qwen_baseline_ko.md
```

### 통과 조건

```yaml
pass_if:
  - Qwen3 baseline completes on selected test subset
  - at least one smaller/quantized Qwen baseline is measured or explicitly deferred
  - memory and latency are measured in a separate clean process
```

---

## M4. LeWM Data Pipeline

### 목표

LeWM 학습용 image/structure pair와 tile/crop pipeline을 안정화한다.

### 작업

```yaml
tasks:
  - preprocess schematic images
  - build global view
  - build tile/crop views
  - encode structure metadata
  - implement augmentations
  - create dataloader sanity check
```

### 권장 구조

```yaml
visual_pipeline:
  global_view:
    resolution: 224 or 336
  tile_view:
    resolution: 224 or 336
  aggregation:
    - attention pooling
    - graph/set pooling
```

### 산출물

```text
results/lewm_data_pipeline/sanity_check.json
docs/20_results/003_lewm_data_pipeline_ko.md
```

### 통과 조건

```yaml
pass_if:
  - one batch can be loaded under RAM 24~26GB
  - image and structure target are aligned
  - tile/crop metadata is traceable back to original sample
```

---

## M5. LeWM-S Structure Pretraining

### 목표

LeWM-S가 회로 image/structure alignment를 학습할 수 있는지 검증한다.

### 초기 모델

```yaml
model:
  name: LeWM-S
  encoder: ViT-tiny
  embed_dim: 192
  predictor_depth: 6
  trainable_parameters: about 18M
```

### Loss

```yaml
losses:
  - image_to_structure_contrastive_alignment
  - masked_crop_or_neighbor_prediction
  - optional_structure_prediction
```

### 측정

```yaml
metrics:
  - train_loss
  - holdout_alignment_retrieval_top1
  - holdout_alignment_retrieval_top5
  - peak_vram_mb
  - throughput
```

### 산출물

```text
checkpoints/lewm_s/latest.pt       # local only
results/lewm_s/pretrain_log.json
docs/20_results/004_lewm_s_pretraining_ko.md
```

### 통과 조건

```yaml
pass_if:
  - training is stable
  - holdout retrieval above random
  - peak VRAM fits within 24GB
```

---

## M6. Frozen Latent Probe

### 목표

LeWM latent가 실제 회로 evidence를 보존하는지 확인한다.

### Probe tasks

```yaml
probe_tasks:
  - component_presence
  - symbol_type
  - structure_retrieval
  - crop_to_component_hit
  - connection_edge_proxy_if_available
```

### 측정

```yaml
metrics:
  - component_presence_f1
  - symbol_type_f1
  - structure_retrieval_top1
  - structure_retrieval_top5
  - crop_to_component_hit
```

### 산출물

```text
results/lewm_s/probe_results.json
docs/20_results/005_lewm_frozen_probe_ko.md
```

### 통과 조건

```yaml
pass_if:
  - at least two probe metrics are above random or trivial baseline
  - failure cases are categorized as perception / structure / tile issue
```

### 중단 조건

```yaml
stop_if:
  - all probe metrics are near random
  - latent collapses across different schematics
  - evidence is lost due to resizing
```

---

## M7. Router Contract and Oracle Baseline

### 목표

LLM router / JEPA router를 학습하기 전에 oracle routing과 registry contract를 고정한다.

### 작업

```yaml
tasks:
  - define taxonomy labels
  - define AdapterCard registry format
  - define oracle adapter id
  - define random router baseline
  - define abstain policy
```

### 산출물

```text
configs/router/adapter_registry.yaml
results/router/oracle_router_baseline.json
docs/20_results/006_router_contract_ko.md
```

### 통과 조건

```yaml
pass_if:
  - oracle router can run
  - random router baseline can run
  - abstain flag is represented
  - uncertified adapters cannot be selected
```

---

## M8. LLM Router Baseline

### 목표

LeWM evidence summary + query + registry를 작은 LLM router가 adapter/taxonomy로 mapping할 수 있는지 본다.

### 입력

```yaml
inputs:
  - user_query
  - visual_evidence_summary
  - component_relation_summary
  - AdapterCard registry summary
```

### 출력

```yaml
outputs:
  - predicted_taxonomy
  - selected_adapter_id
  - abstain_flag
  - route_confidence
```

### 측정

```yaml
metrics:
  - top1_hit
  - routed_score
  - oracle_score
  - random_score
  - abstain_rate
  - router_regret
```

### 산출물

```text
results/router/llm_router_eval.json
docs/20_results/007_llm_router_baseline_ko.md
```

### 통과 조건

```yaml
pass_if:
  - top1_hit > random
  - router_regret is measured
  - abstain_rate is reported
```

---

## M9. JEPA Latent Router Baseline

### 목표

LLM 없이 LeWM latent에서 taxonomy/adapter를 예측할 수 있는지 검증한다.

### 입력

```yaml
inputs:
  - latent_state
  - query_embedding
  - adapter_registry_embedding
```

### 출력

```yaml
outputs:
  - predicted_taxonomy
  - selected_adapter_id
  - abstain_flag
  - route_confidence
```

### 측정

```yaml
metrics:
  - top1_hit
  - oracle_score
  - routed_score
  - random_score
  - router_regret
  - abstain_rate
```

### 산출물

```text
results/router/jepa_router_eval.json
docs/20_results/008_jepa_router_baseline_ko.md
```

### 통과 조건

```yaml
pass_if:
  - top1_hit > random
  - router_regret is measured
  - failure is attributable to perception or router
```

---

## M10. Answer Head / Adapter Head Evaluation

### 목표

LeWM latent를 실제 QA/structure output으로 바꾸는 answer path를 검증한다.

### 비교

```yaml
compare:
  - frozen LeWM + linear head
  - frozen LeWM + MLP head
  - LeWM + answer head LoRA
  - LeWM + structure projector LoRA
```

### 측정

```yaml
metrics:
  - circuit_vqa_score
  - structured_field_match
  - component_probe_score
  - wrong_adapter_damage
```

### 산출물

```text
results/lewm_answer_head/eval.json
docs/20_results/009_answer_head_eval_ko.md
```

### 통과 조건

```yaml
pass_if:
  - LeWM head beats trivial baseline
  - score/resource tradeoff is measurable
  - failure cases are categorized
```

---

## M11. Actual-Only Certification Registry

### 목표

기존 taxonomy-lora-bank의 actual-only 원칙을 LeWM router/answer head에도 적용한다.

### 비교

```yaml
comparisons:
  - base_no_adapter_or_base_head
  - correct_head_or_adapter
  - wrong_head_or_adapter
  - random_untrained_head_or_adapter
```

### actual certified 조건

```yaml
actual_certified_if:
  - correct_score > base_score
  - correct_score > wrong_score + 0.05
  - correct_score > random_score + 0.05
```

### 산출물

```text
configs/adapter_registry/certified_cards.yaml
results/certification/actual_certification_result.json
docs/20_results/010_actual_certification_registry_ko.md
```

### 통과 조건

```yaml
pass_if:
  - no proxy in certification
  - incomplete/actual_failed/actual_certified statuses are used
  - online routing allowed only for certified adapters
```

---

## M12. End-to-End System Comparison

### 목표

최종적으로 Qwen baseline과 LeWM systems를 같은 task subset에서 비교한다.

### 비교군

```yaml
systems:
  - Qwen3-VL single backbone
  - smaller_or_quantized_Qwen
  - LeWM-S + LLM router
  - LeWM-S + JEPA router
  - optional LeWM-M + LLM router
  - optional LeWM-M + JEPA router
```

### 측정

```yaml
metrics:
  - accuracy
  - structured_extraction_score
  - resident_vram_mb
  - peak_vram_mb
  - latency_ms
  - trainable_parameters
  - total_parameters
  - router_regret
  - abstain_rate
```

### 산출물

```text
results/system_comparison/main_table.json
docs/20_results/011_end_to_end_system_comparison_ko.md
```

### 통과 조건

```yaml
strong_success:
  - LeWM system within 5~10pp of Qwen3
  - peak VRAM <= 50~60% of Qwen3
  - failure modes are decomposed

moderate_success:
  - LeWM system beats smaller/low-resource baseline
  - structure probes are strong
  - resource usage is substantially lower than Qwen3

valid_negative:
  - LeWM system is much weaker but failure is attributable
```

---

## M13. LeWM-M Scaling Decision

### 목표

LeWM-S 결과를 보고 더 큰 LeWM으로 갈지 결정한다.

### 조건

```yaml
scale_to_lewm_m_if:
  - LeWM-S probe above random
  - LeWM-S underfits end-to-end task
  - VRAM headroom remains safe

do_not_scale_if:
  - data pipeline is unstable
  - LeWM-S latent collapses
  - probe metrics are random
```

### 후보

```yaml
lewm_m_small:
  params: about 38M
  role: paper candidate if LeWM-S underfits

lewm_m_plus:
  params: about 45M
  role: upper local candidate
```

---

## M14. Paper-Ready Result Table

### 최소 표

```yaml
tables:
  Table_1:
    name: dataset_source_audit
    columns:
      - source
      - role
      - usable_rows
      - license
      - split_policy

  Table_2:
    name: qwen_baselines
    columns:
      - model
      - params
      - score
      - peak_vram
      - resident_vram
      - latency

  Table_3:
    name: lewm_probe
    columns:
      - model
      - params
      - retrieval_top1
      - component_f1
      - symbol_f1
      - peak_vram

  Table_4:
    name: router_eval
    columns:
      - router
      - top1_hit
      - regret
      - abstain_rate
      - wrong_damage

  Table_5:
    name: end_to_end
    columns:
      - system
      - params
      - accuracy
      - peak_vram
      - latency
      - notes
```

### Paper-ready 조건

```yaml
paper_ready_if:
  - Qwen baseline measured
  - LeWM-S trained
  - frozen probe measured
  - at least one router variant measured
  - resource table complete
  - claim boundary includes negative results
```
