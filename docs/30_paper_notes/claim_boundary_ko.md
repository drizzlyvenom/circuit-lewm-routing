# Claim Boundary

Status: paper note draft

---

## 0. Allowed Claims

```yaml
allowed_claims:
  - compact LeWM can be directly compared to Qwen-family VLMs under circuit tasks
  - perception evidence retention can be measured separately from routing and answering
  - actual-only certification can prevent uncertified adapters from being routed
  - resource/accuracy tradeoff can be measured directly
  - large VLM teacher labels can be used as offline training aids only if separated from runtime evaluation
```

---

## 1. Forbidden Claims

```yaml
forbidden_claims:
  - LoRA alone reduces already-loaded Qwen3 backbone VRAM
  - Qwen-trained LoRA transfers to LeWM
  - static circuit image training is identical to physical rollout world modeling
  - proxy metrics certify adapter utility
  - smoke run validates the claim
  - Foveation itself is our main novelty
  - Gemma 4 or another teacher VLM is the final runtime backbone
  - teacher pseudo-labels replace actual-only certification
```

---

## 2. First Paper Spine

```text
negative same-backbone LoRA result
  -> compact perception/world-model hypothesis
  -> circuit dataset source audit
  -> LeWM evidence retention
  -> router comparison
  -> Qwen vs LeWM resource/accuracy table
```

첫 번째 논문형 주장은 Qwen3를 모든 task에서 이기는 것이 아니다. 작은 LeWM이 회로 evidence를 유지하고 resource/accuracy tradeoff를 만들 수 있는지 확인하는 것이다.
