# M2 Circuit Sample Schema and Splits

Status: M2 closed with caveats
Date: 2026-05-29 KST
Manifest script: `scripts/prepare_circuit_samples.py`
Check script: `scripts/check_circuit_curricula.py`

---

## 0. 결론

M2의 목적은 이후 Qwen baseline, LeWM data pipeline, router evaluation이 같은 `CircuitSample` 참조 스키마를 쓰게 만드는 것이다.

이번 단계에서는 원본 이미지/텍스트/정답 payload를 레포에 저장하지 않고, Hugging Face row/file reference와 provenance만 담는 train/holdout/test manifest를 생성했다.

```yaml
outputs:
  train: data/circuit_curricula/train.jsonl
  holdout: data/circuit_curricula/holdout.jsonl
  test: data/circuit_curricula/test.jsonl
  split_summary: data/circuit_curricula/split_summary.json
  check_summary: data/circuit_curricula/check_summary.json
```

M2는 다음 단계로 넘어갈 수 있다. 다만 paper-ready training 전에는 `open-schematics` project/name 중복과 `SchGen_dataset` module 중복을 실제 payload 기준으로 한 번 더 점검해야 한다.

---

## 1. Split Counts

```yaml
total_rows_by_split:
  train: 9216
  holdout: 2176
  test: 2176
```

```yaml
train:
  bshada/open-schematics: 2048
  microsoft/SchGen_dataset: 1024
  ayoubkirouane/CircuitVQA: 4096
  lowercaseonly/cghd: 2048

holdout:
  bshada/open-schematics: 512
  microsoft/SchGen_dataset: 128
  ayoubkirouane/CircuitVQA: 1024
  lowercaseonly/cghd: 512

test:
  bshada/open-schematics: 512
  microsoft/SchGen_dataset: 128
  ayoubkirouane/CircuitVQA: 1024
  lowercaseonly/cghd: 512
```

---

## 2. Split Policy

```yaml
open_schematics:
  policy: deterministic row-reference split over audited train rows
  caveat: project/name duplicate audit remains required before paper-ready training

SchGen_dataset:
  policy: deterministic row-reference split over audited train rows
  caveat: module/schematic duplicate audit remains required before paper-ready training

CircuitVQA:
  policy: provided train split is used for train; provided test split is divided into holdout/test by image row reference
  caveat: license is still not declared on HF

CGHD:
  policy: paired image/XML files are split by drafter+circuit group
  caveat: used as perception probe, not VQA supervision
```

---

## 3. Schema Notes

```yaml
payload_policy:
  raw_images_committed: false
  raw_prompts_committed: false
  raw_expected_answers_committed: false
  signed_asset_urls_committed: false

CircuitVQA:
  prompt: null
  expected_answers: null
  prompt_ref: hf row/text reference
  expected_answer_ref: hf row/text reference
  answer_type: referenced_circuit_vqa_answer
```

`CircuitVQA`의 `answer_type`은 M2에서는 reference-level placeholder다. M3에서 실제 QA payload를 로드할 때 answer normalization과 answer type 분류를 따로 수행한다.

---

## 4. Validation

실행 명령:

```powershell
python .\scripts\prepare_circuit_samples.py
python .\scripts\check_circuit_curricula.py
```

검증 결과:

```yaml
schema_jsonl_valid: true
sample_id_unique: true
qa_answer_type_present: true
structure_refs_present_for_structure_samples: true
cghd_image_xml_refs_present: true
source_group_leakage_across_splits: false
unique_source_groups: 6244
```

Secret/signed URL scan에서는 실제 credential이나 signed asset URL은 발견되지 않았다. `hf_tags` 문서 항목만 `hf_` 패턴에 걸리는 false positive로 확인했다.

---

## 5. M2 Pass Check

```yaml
pass_if:
  train_holdout_test_split_exists: true
  source_leakage_policy_documented: true
  answer_type_present_for_qa_samples: true
  structure_fields_exist_for_structure_pretraining_samples: true

status: closed_with_caveats
next_milestone: M3 Qwen Baseline Measurement
```
