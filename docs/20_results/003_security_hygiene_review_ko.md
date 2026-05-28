# Security Hygiene Review

Status: closed
Date: 2026-05-29 KST
Scope: public repo hygiene after M1-M3

---

## 0. 결론

토큰, API key, signed asset URL, raw CircuitVQA prompt/answer payload는 발견되지 않았다.

다만 공개 레포 위생상 낮은 위험의 local path 노출이 있어 repo-relative path와 placeholder로 정리했다.

---

## 1. Checked Items

```yaml
checked:
  - local absolute Windows paths
  - WSL/Linux absolute paths
  - Hugging Face signed asset URL patterns
  - API key / token / password patterns
  - raw prompt / raw expected answer strings in result artifacts
  - project-persona/internal wording in committed files
```

---

## 2. Findings

```yaml
critical_or_high_findings: none

fixed_low_risk_items:
  - results/qwen/qwen3_single_backbone.json used local absolute paths for model and manifest
  - scripts/run_qwen_baseline.py wrote resolved local paths into result artifacts
  - docs/10_protocols/wsl_lewm_training_boundary_ko.md used a concrete local Windows path as boundary example
  - data/circuit_sources/source_manifest.json included upstream sample path previews from SchGen metadata
```

정리 후 경로는 다음처럼 남긴다.

```yaml
result_artifact_paths:
  manifest: data/circuit_curricula/test.jsonl
  model_path: models/qwen/Qwen3-VL-4B-Instruct

boundary_doc_paths:
  windows_root: <repo-root-on-windows>
  wsl_runtime: <wsl-home>/circuit-lewm-routing

source_manifest_path_preview:
  path_like_values: <path_redacted>/<basename>
```

---

## 3. Residual Risk

```yaml
residual_risk:
  - source_manifest still contains short public dataset field previews for audit reproducibility
  - previous git history contains low-risk generic local path strings from the prior M3 result commit
  - CircuitVQA license remains not declared on HF and is tracked as a dataset/legal caveat, not a secret leak
```

히스토리의 기존 local path 문자열은 Windows user-profile 계열의 일반 로컬 경로라 secret으로 보지 않는다. 그래서 public repo history rewrite는 하지 않는다.

---

## 4. Verification

검증은 다음 범주의 `rg` scan과 JSON/Python syntax check로 진행했다. 이 문서 자체가 secret-pattern false positive를 만들지 않도록 raw token regex는 문서에 그대로 저장하지 않는다.

```yaml
scan_categories:
  - signed URL markers
  - common API key/token/password markers
  - local absolute Windows/WSL path markers
  - raw CircuitVQA prompt/answer substrings
```

검증 결과:

```yaml
secret_scan: no credential or signed URL found
known_false_positives:
  - scripts/audit_sources.py contains sanitizer regex strings for signed URL detection
  - docs use hf_tags as a dataset metadata label
absolute_local_path_scan_after_fix: clean
raw_prompt_answer_scan: clean
json_validation: passed
python_compile: passed
git_diff_check: passed
```
