# 결과 문서 작성 규칙

Status: result brief policy draft

---

## 0. 원칙

결과 문서는 짧아도 다음을 반드시 분리한다.

```yaml
required_sections:
  - what_was_run
  - dataset_and_split
  - metrics
  - result_table
  - failure_modes
  - claim_boundary
  - next_action
```

---

## 1. Active Evidence

다음 조건을 만족한 결과만 active evidence로 둔다.

```yaml
active_evidence_if:
  - source dataset and split are recorded
  - command/config is recorded
  - actual score is measured
  - memory/latency measurement policy is described if resource claim is discussed
  - failure is preserved instead of smoothed over
```

결과 문서 번호는 milestone 번호와 맞춘다. 예를 들어 `M4` 결과는 `004_*.md`로 둔다.

현재 result briefs:

```text
001_dataset_source_audit_ko.md
002_circuit_sample_schema_ko.md
003_qwen_baseline_ko.md
004_lewm_data_pipeline_ko.md
005_lewm_s_pretraining_ko.md
```

---

## 2. Quarantine

다음 결과는 active evidence로 쓰지 않는다.

```yaml
quarantine_if:
  - source or split is unclear
  - proxy result was mixed into certification
  - smoke result is being used as validation evidence
  - data leakage risk is unresolved
  - command/config cannot be reconstructed
```
