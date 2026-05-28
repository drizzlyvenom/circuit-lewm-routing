# M3 Qwen Baseline Measurement

Status: M3 closed with caveats
Date: 2026-05-29 KST
Script: `scripts/run_qwen_baseline.py`

---

## 0. 결론

M3에서는 로컬 `Qwen3-VL-4B-Instruct`를 clean Python process에서 로드하고, M2 `test.jsonl`의 `CircuitVQA` selected subset 64개를 실제 이미지+질문으로 평가했다.

```yaml
outputs:
  qwen3_single_backbone: results/qwen/qwen3_single_backbone.json
  smaller_or_quantized_qwen: results/qwen/qwen_small_or_quantized.json
```

작은/양자화 Qwen 비교군은 M3 시점에 `models/qwen` 아래 로컬 checkpoint가 없어서 명시적으로 deferred 처리했다.

---

## 1. What Was Run

실행 환경:

```yaml
python_env: .venv
torch: 2.11.0+cu128
transformers: 5.9.0
datasets: 4.8.5
gpu: NVIDIA GeForce RTX 3090
clean_process: true
```

모델:

```yaml
name: Qwen3-VL-4B-Instruct
local_path: models/qwen/Qwen3-VL-4B-Instruct
dtype: bfloat16
local_files_only: true
total_parameters: 4437815808
```

명령:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_qwen_baseline.py --max-samples 64 --checkpoint-every 8 --output .\results\qwen\qwen3_single_backbone.json --deferred-small-output .\results\qwen\qwen_small_or_quantized.json
```

---

## 2. Dataset and Split

```yaml
manifest: data/circuit_curricula/test.jsonl
source_dataset: ayoubkirouane/CircuitVQA
source_split: test
selected_vqa_samples: 64
selection_policy: first N vqa_evaluation rows from M2 test manifest
license_caveat: HF dataset card does not declare a license
```

결과 JSON에는 원본 prompt, expected answer, image payload를 저장하지 않는다. Sample id, row reference, question index, answer/prediction hash, latency, correctness만 저장한다.

---

## 3. Metrics

```yaml
circuit_vqa_score_contains_normalized: 0.4375
correct_contains_normalized: 28
sample_count: 64

latency_ms_mean: 691.297
latency_ms_p50: 574.195
latency_ms_p95: 1285.723

vram_before_load_mb: 1682
resident_vram_after_load_mb: 10378
max_resident_vram_observed_mb: 10962
peak_torch_allocated_mb: 8794.849
peak_torch_reserved_mb: 8932.0
```

점수는 expected answer의 `Answer:` 뒤 짧은 정답과 모델 출력을 normalize한 뒤 containment로 비교한 값이다. 따라서 이 값은 M3용 baseline health/resource metric이지, 아직 paper-ready CircuitVQA 공식 점수가 아니다.

---

## 4. Smaller / Quantized Baseline

```yaml
status: deferred
reason: No smaller or quantized Qwen VLM checkpoint is present under local models/qwen at M3 time.
```

이 항목은 M3 통과 조건의 "measured or explicitly deferred" 중 deferred로 닫았다. 추후 작은/양자화 Qwen VLM을 받으면 같은 selected subset과 같은 script schema로 재측정한다.

---

## 5. Claim Boundary

이번 결과로 말할 수 있는 것:

```yaml
safe_claims:
  - local Qwen3-VL-4B-Instruct baseline runs on RTX 3090 in a clean process
  - selected 64 CircuitVQA test samples are evaluated with actual image+question payloads
  - resident/peak VRAM and latency are recorded with reproducible command/config
```

이번 결과로 아직 말하면 안 되는 것:

```yaml
unsafe_claims:
  - full 1024-sample CircuitVQA test accuracy
  - official CircuitVQA benchmark score
  - LeWM system is better or worse than Qwen3
  - smaller/quantized Qwen resource tradeoff
```

---

## 6. M3 Pass Check

```yaml
pass_if:
  qwen3_baseline_completes_on_selected_test_subset: true
  smaller_or_quantized_qwen_measured_or_explicitly_deferred: true
  memory_and_latency_measured_in_separate_clean_process: true

status: closed_with_caveats
next_milestone: M4 LeWM Data Pipeline
```
