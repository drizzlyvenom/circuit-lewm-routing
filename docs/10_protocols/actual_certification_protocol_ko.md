# Actual-Only Certification Protocol

Status: protocol draft
Scope: LeWM router / answer head / adapter certification

---

## 0. 원칙

```text
No proxy in certification.
No actual eval, no certification.
Failure is a result.
No smoke result as validation evidence.
```

Teacher, rule, heuristic, proxy score는 curriculum이나 diagnostic에는 쓸 수 있지만 certification에는 쓸 수 없다.

---

## 1. 비교 조건

동일 holdout samples에서 다음을 actual로 비교한다.

```yaml
actual_certification:
  comparisons:
    - base_no_adapter_or_base_head
    - correct_lora_or_head
    - wrong_lora_or_head
    - random_untrained_lora_or_head

  required:
    - same_holdout_samples
    - same_perception_model
    - same_visual_policy
    - same_roi_or_tile_policy
    - same_answer_normalization
```

---

## 2. Score Fields

```yaml
scores:
  base_score: float
  correct_score: float
  wrong_score: float
  random_score: float

margins:
  gain_vs_base: correct_score - base_score
  margin_vs_wrong: correct_score - wrong_score
  margin_vs_random: correct_score - random_score
```

---

## 3. Status

```yaml
status:
  incomplete:
    meaning: actual comparison is missing

  actual_failed:
    meaning: actual comparison completed but gates failed

  actual_certified:
    meaning: correct adapter/head passed actual gates
```

Certification gate:

```yaml
actual_certified_if:
  - correct_score > base_score
  - correct_score > wrong_score + 0.05
  - correct_score > random_score + 0.05
```

---

## 4. Router Safety

```yaml
routing_allowed_if:
  - actual_certification.status == actual_certified

if_uncertified:
  online_routing_allowed: false
```

Router confidence가 낮으면 adapter를 고르지 않는다.

```yaml
if route_confidence < threshold:
  selected_adapter_id: null
  route: base_or_abstain
```

---

## 5. 금지

```yaml
forbidden_in_certification:
  - deterministic difficulty proxy
  - base_score - epsilon wrong score
  - random_score = base_score
  - mixed_actual_proxy_score
  - teacher_label_as_final_truth
  - proxy_certified
  - smoke_validated
```
