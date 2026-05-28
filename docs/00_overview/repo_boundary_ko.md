# 레포 경계와 분리 판단

Status: repo boundary draft
Scope: 기존 taxonomy-lora-bank repo와 새 Circuit LeWM repo의 역할 분리

---

## 0. 결론

```yaml
decision: split_repo_recommended
```

기존 `taxonomy-lora-bank-for-vision-model` 레포는 protocol/control-plane reference로 남기고, 이 레포는 회로 도메인 Perception LeWM 학습과 비교 실험을 맡는다.

---

## 1. 기존 레포의 역할

```yaml
taxonomy_lora_bank_repo:
  role:
    - actual-only AdapterCard / certification protocol
    - online/offline LoRA registry/compiler control plane
    - prior Qwen same-backbone negative evidence archive
    - Qwen reference certification utilities
```

기존 레포의 부정 결과는 이 실험의 출발점이다. 같은 Qwen backbone이 이미 resident 상태라면 LoRA adapter만으로 backbone VRAM 자체를 줄일 수 없다는 경계가 생겼다.

---

## 2. 새 레포의 역할

```yaml
new_circuit_lewm_repo:
  role:
    - circuit-domain Perception LeWM training
    - open-schematics / SchGen / CircuitVQA / CGHD data pipeline
    - Qwen baseline vs LeWM comparison
    - LLM router vs JEPA router comparison
    - resource/accuracy tradeoff evaluation
```

분리 이유:

```yaml
why_split:
  - LeWM training은 dataset/checkpoint/architecture 중심이고, 기존 repo는 registry/certification 중심임
  - 회로 domain data audit, pretraining, latent probe, router head는 별도 lifecycle을 가짐
  - 기존 Qwen same-backbone LoRA negative result와 새 LeWM hypothesis를 섞으면 claim이 흐려짐
  - 새 repo는 대용량 local artifacts와 학습 로그가 많아질 가능성이 높음
  - README와 milestone의 주어가 완전히 달라짐
```

---

## 3. 아직 나누지 않는 것

```yaml
do_not_split_yet:
  - LLM router repo
  - JEPA router repo
  - dataset-only repo
  - AdapterCard library repo
```

초기에는 이 레포 하나에서 회로 LeWM, router, answer head, comparison table을 같이 다룬다. 재사용이 반복되면 나중에 `taxonomy_lora_core` 또는 `circuit_lewm_core` 같은 package 분리를 검토한다.

---

## 4. 공유 contract

기존 repo에서 개념적으로 가져오는 것은 weight가 아니라 protocol이다.

```yaml
shared_contracts:
  - AdapterCard
  - ActualCertificationResult
  - RouteTrace
  - actual-only certification rule
  - claim boundary rules
```

초기에는 schema를 복사해 시작한다. 실제 코드 package 공유는 나중에 반복 사용이 확인된 뒤 결정한다.
