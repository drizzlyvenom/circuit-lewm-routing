# M2 Circuit Sample Schema and Splits

Status: M2 closed
Date: 2026-05-29 KST
Script: `scripts/prepare_circuit_samples.py`
Summary: `data/circuit_curricula/usable_pair_summary.json`

---

## 0. What Was Run

M2에서는 후속 Qwen baseline, LeWM data pipeline, LeWM pretraining, router evaluation이 공통으로 참조할 `CircuitSample` JSONL curriculum을 만들었다.

실행 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_circuit_samples.py
```

산출물:

```text
data/circuit_curricula/train.jsonl
data/circuit_curricula/holdout.jsonl
data/circuit_curricula/test.jsonl
data/circuit_curricula/usable_pair_summary.json
```

---

## 1. Dataset and Split

```yaml
total_written:
  train: 15208
  holdout: 1637
  test: 2942

by_source:
  bshada/open-schematics:
    train: 5000
    holdout: 512
    test: 512
  lowercaseonly/cghd:
    train: 2690
    holdout: 267
    test: 336
  ayoubkirouane/CircuitVQA:
    train: 7518
    holdout: 858
    test: 2094
```

`open-schematics`는 실제 image object의 width/height가 양수이고 `schematic` 문자열이 비어 있지 않은 row만 curriculum에 넣었다.

`CGHD`는 로컬 ignored 다운로드 경로인 `data/downloads/hf/lowercaseonly_cghd`에서 image/XML stem을 맞춰 pair를 만들었다.

`CircuitVQA`는 프롬프트와 정답 본문을 커밋하지 않고 HF row reference와 `answer_type: hf_conversation_ref`만 기록했다.

---

## 2. Metrics

```yaml
open_schematics:
  scanned_rows: 6349
  verified_pairs:
    train: 5000
    holdout: 512
    test: 512
  failed_rows:
    missing_or_bad_image: 24

cghd:
  image_files: 4181
  xml_annotation_files: 3293
  paired_image_xml_files: 3293
  bad_images: 0
  unique_split_groups: 1043

circuitvqa:
  verified_source_rows:
    train: 8376
    test: 2094
  prompt_answer_payload_committed: false
```

---

## 3. Result Table

| Check | Result |
|---|---:|
| train/holdout/test files exist | true |
| `open-schematics` train pairs >= 5k | true |
| `open-schematics` rows verified before inclusion | true |
| leakage policy documented | true |
| QA samples have `answer_type` | true |
| structure pretraining samples have structure refs | true |

M2 status는 `closed`다.

---

## 4. Failure Modes

`open-schematics`에서 24개 row는 image decode 또는 width/height 검증에 실패해 제외했다.

`CGHD`는 XML annotation이 있는 3,293개 image/XML pair만 사용했다. image만 있고 XML이 없는 파일은 perception probe supervision으로 쓰지 않았다.

`CircuitVQA`는 HF에 license가 선언되어 있지 않으므로 prompt/answer 본문은 커밋하지 않았다.

---

## 5. Claim Boundary

이번 M2는 dataset split과 reference curriculum을 닫는 작업이다. LeWM 학습 성능, Qwen baseline accuracy, VRAM 절감 claim은 아직 active evidence가 아니다.

현재 주장할 수 있는 것은 다음뿐이다.

```yaml
safe_claims:
  - open-schematics에서 5k train image+schematic verified pair curriculum을 만들었다
  - CGHD local snapshot에서 3293 image/XML pair를 만들었다
  - CircuitVQA는 prompt/answer payload 없이 row reference만 curriculum에 넣었다
  - train/holdout/test split과 source별 leakage policy를 기록했다
```

---

## 6. Next Action

M3에서는 이 curriculum의 `CircuitVQA` test reference를 사용해 Qwen3-VL single backbone과 smaller/quantized Qwen baseline을 같은 split에서 측정한다.

M4에서는 `open-schematics` structure pair와 `CGHD` annotation pair를 사용해 LeWM data pipeline이 image/structure alignment와 tile/crop metadata를 제대로 유지하는지 검증한다.
