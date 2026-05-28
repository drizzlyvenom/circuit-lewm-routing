# M5b ROI-aware Structure Target Audit

Status: redesign audit completed, user review pending
Measured at: 2026-05-29 KST

---

## what_was_run

M5 첫 5k `LeWM-S` 학습에서 holdout retrieval이 random top1을 넘지 못했기 때문에, 학습량을 늘리기 전에 structure target 자체가 충분한지 감사했다.

이번 감사의 목적은 `component multi-hot`만으로 LeWM backbone을 학습시키는 대신, KiCad schematic에서 파싱한 graph/set target과 ROI trace target을 쓸 수 있는지 확인하는 것이다.

```yaml
script: scripts/audit_roi_structure_targets.py
output: results/structure_targets/roi_structure_target_audit.json
train_manifest: data/circuit_curricula/train.jsonl
holdout_manifest: data/circuit_curricula/holdout.jsonl
open_train_limit: 5000
open_holdout_limit: 512
cghd_train_limit: 2690
component_vocab_size: 4096
```

출력 정책:

```yaml
stores_raw_images: false
stores_raw_schematic_text: false
stores_raw_xml: false
stores_text_hashes_only_for_labels: true
```

---

## dataset_and_split

```yaml
open_schematics:
  source_dataset: bshada/open-schematics
  source_url: https://huggingface.co/datasets/bshada/open-schematics
  role:
    - KiCad schematic parse audit
    - image-to-structure graph/set target feasibility
    - ROI trace and wire skeleton feasibility
  train_records: 5000
  holdout_records: 512

cghd:
  source_dataset: lowercaseonly/cghd
  source_url: https://huggingface.co/datasets/lowercaseonly/cghd
  role:
    - ROI box supervision audit
    - symbol/text/junction detail specialist source
  train_records: 2690
```

`open-schematics`는 HF streaming으로 schematic text만 읽고, image decode는 하지 않았다. `CGHD`는 M2 manifest가 가리키는 ignored local XML을 읽어 box 통계만 남겼다.

---

## metrics

`open-schematics` train 5k 기준 collision은 다음과 같다.

```yaml
component_multihot:
  records: 5000
  unique_target_count: 4779
  duplicate_sample_ratio: 0.069
  largest_collision_group: 10

parsed_symbol_lib_count:
  records: 5000
  unique_target_count: 4814
  duplicate_sample_ratio: 0.06
  largest_collision_group: 7

roi_graph_signature:
  records: 5000
  unique_target_count: 4830
  duplicate_sample_ratio: 0.0548
  largest_collision_group: 5
```

`open-schematics` holdout 512 기준 collision은 다음과 같다.

```yaml
component_multihot:
  records: 512
  unique_target_count: 510
  duplicate_sample_ratio: 0.0078125

roi_graph_signature:
  records: 512
  unique_target_count: 511
  duplicate_sample_ratio: 0.00390625
```

KiCad parse 가능성:

```yaml
train_records_parsed: 5000
samples_with_symbols: 5000
samples_with_wires: 4896
samples_with_labels: 4086
samples_with_roi_trace: 5000
samples_with_wire_skeleton: 4896
mean_symbol_count: 53.5286
mean_wire_count: 118.2382
mean_roi_candidate_count: 123.614
mean_tile_4x4_symbol_occupancy: 11.4674
```

CGHD ROI supervision:

```yaml
records: 2690
samples_with_boxes_ratio: 1.0
empty_or_bad_xml_records: 0
mean_boxes_per_sample: 76.3413
median_boxes_per_sample: 48
box_count_range: [6, 542]
mean_roi_area_ratio: 0.0012705
median_roi_area_ratio: 0.0003755
top_labels:
  - text
  - junction
  - resistor
  - terminal
  - crossover
```

---

## result_table

| check | result | note |
|---|---:|---|
| KiCad schematic parse available | true | 5,000/5,000 train records parsed |
| ROI trace target available | true | symbol position and schematic bbox exist for 5,000/5,000 |
| wire skeleton target available | true | 4,896/5,000 train records have wires |
| graph/set target better than component multihot | true | train duplicate ratio `0.0548` vs `0.069` |
| CGHD ROI box supervision available | true | 2,690/2,690 records have boxes |
| longer M5 training ready | false | target redesign should be tested before another 5k+ run |
| M5.3 diagnostic ready | true | 512/128 diagnostic run can be prepared |

---

## failure_modes

- `component multi-hot`는 같은 component set을 가진 서로 다른 회로를 구분하지 못한다. train 5k에서 duplicate sample ratio가 `0.069`로 남았다.
- `roi_graph_signature`도 완전한 netlist equivalence target은 아니다. wire count, label count, junction count, tile occupancy를 포함한 구조 서명이므로 topology proxy에 가깝다.
- KiCad parser는 이번 audit에 필요한 top-level `symbol`, `wire`, `label`, `junction`, `no_connect`만 읽는다. hierarchical sheet, net class, pin-level connectivity는 아직 target에 포함하지 않았다.
- CGHD box는 ROI/detail supervision으로는 좋지만, schematic-level graph target을 대신하지는 못한다.
- HF Hub 비인증 경고가 출력됐다. 실행은 성공했지만 반복 실행이 많아지면 cache/token 전략을 별도로 정리하는 편이 안전하다.

---

## claim_boundary

이번 감사로 말할 수 있는 것은 다음까지다.

- `open-schematics` KiCad schematic에서 LeWM용 graph/set target과 ROI trace target을 만들 수 있다.
- `component multi-hot` 단독 target보다 `roi_graph_signature` 쪽이 collision이 낮다.
- CGHD는 detail specialist 또는 tile probe용 ROI box supervision으로 쓸 수 있다.
- LoRA/전문가 모듈이 ROI detail을 잡는 초기 설계는 유지하는 편이 맞다.

아직 말할 수 없는 것은 다음과 같다.

- `roi_graph_signature` target으로 학습하면 LeWM-S retrieval이 random을 넘는다는 주장.
- pin-level connectivity나 회로 기능 이해가 이미 학습 가능하다는 주장.
- M5를 닫아도 된다는 주장.
- Qwen3 baseline과 end-to-end accuracy/resource 비교를 시작해도 된다는 주장.

---

## next_action

다음 실행은 긴 5k+ 학습이 아니라 M5.3 diagnostic run으로 둔다.

```yaml
recommended_next:
  name: M5.3 ROI-aware graph/set target diagnostic
  train_records: 512
  holdout_records: 128
  backbone_role: roi_aware_structure_backbone
  target:
    - parsed symbol family/lib set
    - symbol position/tile occupancy
    - wire/junction/label count and skeleton proxy
    - ROI candidate trace
  preserve_lora_roi_detail_design: true
  specialist_detail_sources:
    - CGHD ROI boxes
    - later CircuitVQA or circuit-specific QA supervision
  success_signal:
    - holdout retrieval clearly above random
    - ROI/tile probe above trivial baseline
    - no raw payload committed
```

즉, LeWM 백본은 회로의 구조 골격과 ROI 후보를 보존하는 역할까지만 맡기고, 작은 글자, 정확한 symbol class, OCR, pin/value 같은 디테일은 LoRA/전문가 경로에 남긴다.
