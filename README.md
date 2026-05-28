# Circuit LeWM Routing

Status: M5 first 5k LeWM-S run recorded with caveats
Scope: circuit-domain Perception LeWM training + router/adapter evaluation
Target hardware: RTX 3090 24GB VRAM, local RAM 24~26GB
Primary question: compact world-model perception vs monolithic VLM backbone

이 레포는 회로 도메인에서 **작은 Perception LeWM이 회로 구조 evidence를 충분히 보존하도록 학습될 수 있는지** 검증하기 위한 실험 레포다.

기존 `taxonomy-lora-bank-for-vision-model` 레포는 Qwen 계열 VLM 위에서 taxonomy LoRA bank와 actual-only certification protocol을 검증하던 control-plane 성격의 레포였다. 이 레포는 그 다음 단계로, 같은 문제를 더 작은 perception/world-model backbone으로 옮겨 검증한다.

LeWM 학습은 Windows/Codex 작업 디렉터리에서 직접 돌리지 않고, WSL/Linux 전용 영역에서 실행한다. Windows 쪽 루트는 문서, 스키마, source audit, Qwen baseline, router/eval orchestration을 맡고, world-model training runtime은 `wsl/lewm/` 아래로 분리한다.

## 핵심 질문

```text
Perception LeWM이 회로 구조 evidence를 작은 latent에 충분히 보존한다면,
작은 router / answer head / certified adapter만으로
단일 Qwen3-VL backbone 대비 더 낮은 resident/peak VRAM에서
허용 가능한 회로 QA 및 structure extraction 성능을 낼 수 있는가?
```

중요한 주장 경계:

- LoRA만으로 이미 로드된 Qwen3 backbone VRAM을 줄인다고 주장하지 않는다.
- Qwen에서 학습한 LoRA weight가 LeWM으로 transfer된다고 주장하지 않는다.
- static circuit image training을 physical rollout world modeling과 동일하다고 주장하지 않는다.
- proxy metric으로 certification을 대체하지 않는다.
- validation에는 smoke result를 쓰지 않는다.

## 비교군

```text
B0. Qwen3-VL single backbone
B1. smaller or quantized Qwen baseline
E1. Perception LeWM + LoRA Routing LLM
E2. Perception LeWM + JEPA latent router
E3. Perception LeWM + simple non-LoRA head ablation
```

핵심 비교는 `B0 / E1 / E2`다. `B1`은 저자원 monolithic baseline이고, `E3`은 LeWM latent가 실제 회로 evidence를 담는지 보는 ablation이다.

## 데이터셋 역할

```yaml
structure_pretraining:
  - bshada/open-schematics
  - microsoft/SchGen_dataset

vqa_evaluation:
  - ayoubkirouane/CircuitVQA
  - MirandaAbhilash/circuitvqa-dataset optional

perception_probe:
  - lowercaseonly/cghd
```

`open-schematics`는 schematic image와 KiCad/components metadata를 연결하는 핵심 source로 둔다. `SchGen_dataset`은 image pretraining source라기보다 circuit structure/text/code prior로 사용한다.

## 첫 모델 크기

로컬 LeWM config 기준 첫 모델은 `LeWM-S`로 둔다.

```yaml
LeWM-S:
  encoder: ViT-tiny
  embed_dim: 192
  predictor_depth: 6
  estimated_trainable_parameters: about 18M

LeWM-M-small:
  encoder: ViT-small
  embed_dim: 384
  predictor_depth: 4
  estimated_trainable_parameters: about 38M
```

처음부터 큰 모델로 가지 않는다. `LeWM-S`가 latent probe에서 signal을 보이고 end-to-end task에서 underfit일 때만 `LeWM-M`으로 확장한다.

## 문서 구조

```text
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
    claim_boundary_ko.md

schemas/
configs/
data/
results/
src/circuit_lewm/
scripts/
wsl/lewm/
```

## 현재 단계

현재는 M5 첫 5k LeWM-S run을 기록한 단계다. Qwen3-VL 단일 backbone baseline, LeWM data pipeline sanity, LeWM-S 5k 학습 health 결과는 active evidence로 추가됐다. 다만 M5 첫 run의 최종 holdout retrieval은 random top1을 넘지 못해 구조 target/encoder 재설계가 필요하다.

첫 실행 순서는 다음과 같다.

```text
M0. Repo scaffold and boundary
M1. Dataset source audit                    closed_with_caveats
M2. Usable circuit sample schema and splits closed
M3. Qwen baseline measurement              closed
M4. LeWM data pipeline                     closed
M5. LeWM-S structure pretraining           first_run_closed_with_caveats
```

자세한 마일스톤은 [circuit_lewm_validation_milestones_ko.md](docs/10_protocols/circuit_lewm_validation_milestones_ko.md)를 따른다.

M1 결과 문서는 [001_dataset_source_audit_ko.md](docs/20_results/001_dataset_source_audit_ko.md)에 있다.

M2 결과 문서는 [002_circuit_sample_schema_ko.md](docs/20_results/002_circuit_sample_schema_ko.md)에 있고, split manifest는 `data/circuit_curricula/`에 있다.

M3 결과 문서는 [003_qwen_baseline_ko.md](docs/20_results/003_qwen_baseline_ko.md)에 있다.

M4 결과 문서는 [004_lewm_data_pipeline_ko.md](docs/20_results/004_lewm_data_pipeline_ko.md)에 있다.

M5 첫 5k 결과 문서는 [005_lewm_s_pretraining_ko.md](docs/20_results/005_lewm_s_pretraining_ko.md)에 있다.
