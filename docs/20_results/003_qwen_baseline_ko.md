# M3 Qwen Baseline Result

Status: M3 closed
Measured at: 2026-05-29 KST

---

## what_was_run

M3에서는 M2에서 만든 `CircuitVQA` test reference curriculum을 사용해 `Qwen3-VL-4B-Instruct` 단일 backbone baseline을 별도 clean process에서 측정했다.

```yaml
baseline:
  id: qwen3_vl_single
  model_path: models/qwen/Qwen3-VL-4B-Instruct
  dtype: bfloat16
  device_map: auto
  script: scripts/run_qwen_baseline.py
  output: results/qwen/qwen3_single_backbone.json
```

작은 Qwen 또는 quantized Qwen baseline은 현재 로컬 `models/qwen` 아래에 사용할 checkpoint가 없고, Windows venv에 `bitsandbytes`가 없어 M3에서는 명시 보류했다.

```yaml
deferred_baseline:
  id: smaller_or_quantized_qwen
  output: results/qwen/qwen_small_or_quantized.json
  reason: no local smaller_or_quantized_qwen_checkpoint_and_no_bitsandbytes
```

---

## dataset_and_split

```yaml
source_dataset: ayoubkirouane/CircuitVQA
source_url: https://huggingface.co/datasets/ayoubkirouane/CircuitVQA
license: not_declared_on_hf
source_split: test
curriculum: data/circuit_curricula/test.jsonl
selected_subset_policy: first deterministic CircuitVQA test QA refs from M2 curriculum
qa_pairs: 64
prompt_answer_payload_committed: false
prediction_payload_committed: false
```

원문 prompt, 정답 본문, 모델 예측문은 레포에 커밋하지 않는다. 결과 JSON에는 sample reference, 정답 여부, latency, 문자열 길이만 남긴다.

---

## metrics

`circuit_vqa_score`는 normalized relaxed match로 기록했다. `structured_extraction_score_if_available`은 이번 M3에서 측정하지 않았다.

```yaml
qwen3_vl_single:
  qa_pairs: 64
  circuit_vqa_score: 0.484375
  normalized_relaxed_match: 0.484375
  normalized_exact_match: 0.1875
  relaxed_hits: 31
  exact_hits: 12
  total_parameters: 4437815808
  latency_ms_mean: 737.31
  latency_ms_min: 171.69
  latency_ms_max: 2435.87
  load_seconds: 5.23
  process_ram_mb_after_eval: 1932.633
```

VRAM은 두 계열로 남긴다.

```yaml
vram_measurement:
  process_specific_nvidia_smi_query:
    resident_vram_mb_after_load: 0
    resident_vram_mb_after_eval: 0
    note: Windows에서 query-compute-apps 기반 process별 사용량이 잡히지 않았다.
  whole_gpu_delta:
    gpu_used_vram_mb_before_load: 1677
    gpu_used_vram_mb_after_load: 10325
    gpu_used_vram_mb_after_eval: 11131
    resident_vram_mb_after_load_delta: 8648
    resident_vram_mb_after_eval_delta: 9454
  torch_peak:
    peak_vram_mb_torch_allocated: 8888.74
    peak_vram_mb_torch_reserved: 9218.0
```

---

## result_table

| baseline | status | QA pairs | score | exact | resident VRAM evidence | torch peak allocated | mean latency | params | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-VL 4B single | completed | 64 | 0.4844 | 0.1875 | 9454 MB whole-GPU delta after eval | 8888.74 MB | 737.31 ms | 4.44B | strong monolithic baseline |
| smaller/quantized Qwen | deferred | 0 | N/A | N/A | N/A | N/A | N/A | N/A | no local checkpoint or quantized runtime |

---

## failure_modes

- 작은/양자화 Qwen baseline은 아직 실제 측정값이 아니다.
- Windows 환경에서 `nvidia-smi --query-compute-apps` 기반 process별 resident VRAM이 0으로 반환되어, 이번 문서에서는 whole-GPU 전후 차분과 torch peak를 함께 쓴다.
- 이번 score는 문자열 normalized exact/relaxed match이며, 의미 기반 VQA judge나 구조 추출 점수와 동일하지 않다.
- 평가 범위는 `CircuitVQA` test reference 64개 QA pair이며, 전체 test split 대표성은 아직 별도 검증하지 않았다.

---

## claim_boundary

이번 M3의 active evidence로 말할 수 있는 것은 다음까지다.

- 로컬 RTX 3090에서 `Qwen3-VL-4B-Instruct` 단일 backbone baseline이 M2 `CircuitVQA` test reference 64개 QA pair에서 완료됐다.
- 같은 run에서 normalized relaxed score는 `0.484375`, normalized exact score는 `0.1875`였다.
- resource evidence는 torch peak와 whole-GPU VRAM delta 기준으로 남겼다.
- 작은/양자화 Qwen baseline은 M3 통과 조건에 따라 명시 보류됐으며, 성능 비교값으로 사용하지 않는다.

아직 말할 수 없는 것은 다음과 같다.

- LeWM이 Qwen3보다 accuracy가 높거나 낮다는 주장.
- LeWM이 Qwen3 대비 VRAM을 줄인다는 주장.
- 이 64개 QA pair 결과가 전체 회로 VQA 성능을 대표한다는 주장.
- 문자열 relaxed match가 회로 구조 이해를 완전히 측정한다는 주장.

---

## next_action

M4에서는 같은 split 경계를 유지한 채 LeWM data pipeline을 만든다. 이후 M5/M6에서 LeWM-S 학습과 probe/eval을 닫은 뒤, M10/M11에서 이번 Qwen baseline과 resource/accuracy tradeoff를 비교한다.
