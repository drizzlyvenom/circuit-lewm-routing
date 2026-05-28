# Offline Teacher Loop Protocol

Status: M5 follow-up design draft
Scope: Gemma 4 offline teacher as structure-anchor generator, not runtime backbone

---

## 0. 목적

M5와 M5.3의 병목은 LeWM-S 학습량보다 **image latent가 따라갈 안정적인 structure anchor가 약하다**는 점에 가깝다.

이 문서는 Gemma 4 같은 큰 VLM을 runtime system에 넣기 위한 문서가 아니다. 큰 teacher model은 offline loop에서만 쓰고, 최종 비교 시스템은 여전히 compact LeWM backbone과 router/adapter 구조로 둔다.

```yaml
teacher_role:
  - offline annotator
  - offline critic
  - pseudo-label consistency checker
  - hard-case miner

not_teacher_role:
  - final runtime backbone
  - answer oracle
  - unrestricted free-text target generator
  - certification replacement
```

---

## 1. 핵심 원칙

Teacher 출력은 자유문장 imitation으로 쓰지 않는다.

```yaml
allowed_teacher_outputs:
  - roi_tiles
  - component_family_candidates
  - junction_or_wire_density
  - visible_text_presence
  - uncertainty_flags
  - hard_case_reason_code

forbidden_teacher_outputs:
  - chain_of_thought
  - unconstrained long explanation
  - final answer used as ground truth
  - labels without evidence source
```

Teacher label은 항상 source evidence와 함께 검증한다.

```yaml
evidence_sources:
  - KiCad parsed schematic fields
  - CGHD XML boxes
  - deterministic image/tile coordinates
  - existing Qwen baseline result for comparison only
```

---

## 2. 1차 검증 루프

Gemma 4를 바로 5k pseudo-labeler로 쓰지 않는다. 먼저 128개 내외에서 teacher가 안정적인지 본다.

```yaml
teacher_audit_128:
  input:
    - schematic image
    - optional parsed KiCad summary
    - optional CGHD box summary if source is CGHD
  output_schema:
    roi_tiles: list[int]
    component_family_candidates: list[str]
    wire_density_level: low | medium | high
    junction_density_level: low | medium | high
    visible_text_level: none | sparse | dense
    uncertainty_flags: list[str]
  checks:
    - schema_valid_json
    - roi_tiles_inside_grid
    - component_family_overlap_with_kicad
    - density_consistent_with_kicad_counts
    - uncertainty_rate_reported
```

통과 조건은 teacher가 완벽한 정답을 내는 것이 아니다. LeWM target으로 쓸 수 있을 만큼 **일관되고 검증 가능한 보조 label**을 내는지 확인하는 것이다.

---

## 3. LeWM 학습에 쓰는 방식

Teacher output은 compact vector target으로 변환한다.

```yaml
lewm_teacher_target:
  global_structure:
    - component_family_presence
    - wire_density_bucket
    - junction_density_bucket
    - visible_text_bucket
  roi_structure:
    - roi_tile_multilabel
    - high_uncertainty_tile_mask
  sample_weight:
    - downweight_if_teacher_uncertain
    - downweight_if_kicad_disagrees
```

학습 objective는 다음처럼 분리한다.

```yaml
objectives:
  fixed_teacher_target_prediction:
    role: stable anchor
  image_to_fixed_target_retrieval:
    role: representation alignment
  roi_tile_prediction:
    role: perception localization
  crop_prediction:
    role: local evidence retention
```

중요한 점은 teacher label을 만든 뒤 LeWM 학습 중에는 target이 고정되어야 한다는 것이다. M5.3처럼 image encoder와 structure encoder가 동시에 움직이는 구조는 정렬 기준이 흔들릴 수 있다.

---

## 4. 결과 해석 경계

Offline teacher loop로 말할 수 있는 것은 다음까지다.

- 큰 VLM을 offline으로 써서 compact LeWM target을 더 안정적으로 만들 수 있는지.
- teacher pseudo-label이 KiCad/CGHD evidence와 어느 정도 일치하는지.
- fixed target을 썼을 때 LeWM retrieval/probe가 M5.3보다 좋아지는지.

아직 말할 수 없는 것은 다음과 같다.

- Gemma 4가 회로 QA 정답 oracle이라는 주장.
- teacher pseudo-label만으로 actual certification을 대체할 수 있다는 주장.
- teacher가 생성한 자유문장을 LeWM이 이해했다고 주장하는 것.
- runtime system이 Gemma 4를 사용한다고 주장하는 것.

---

## 5. 다음 실험 후보

```yaml
M5_4_teacher_anchor_audit:
  train_records: 128
  holdout_records: 64
  goal:
    - validate Gemma 4 teacher schema outputs
    - compare teacher labels against KiCad/CGHD-derived counters
    - decide whether pseudo-labeling is worth scaling

M5_5_fixed_teacher_target_diagnostic:
  train_records: 512
  holdout_records: 128
  goal:
    - train LeWM-S against fixed teacher/KiCad hybrid target
    - report train retrieval, holdout retrieval, ROI tile probe, and target prediction
```

M5.4가 불안정하면 teacher loop를 확대하지 않는다. 그 경우에는 KiCad parser를 더 강한 netlist/topology target으로 개선하는 쪽이 우선이다.
