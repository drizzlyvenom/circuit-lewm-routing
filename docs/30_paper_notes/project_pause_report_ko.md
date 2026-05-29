# 프로젝트 잠정 중단 보고서 초안

Status: draft for README merge
Date: 2026-05-29
Scope: Circuit LeWM Routing / compact perception world-model exploration

---

## 0. 요약

본 프로젝트는 회로 도메인에서 compact Perception LeWM backbone, taxonomy/router, ROI routing, LoRA/adapter bank를 함께 검증하는 방향으로 시작했다.

M1부터 M5.3까지 진행한 결과, 데이터셋 수집, split 구성, Qwen baseline, WSL 기반 LeWM-S 학습 실행은 가능했다. 그러나 M5와 M5.3의 결과는 LeWM latent가 회로 구조 evidence를 안정적으로 보존한다고 보기에는 부족했다.

핵심 문제는 단순 학습량 부족이라기보다, **image latent가 따라갈 structure anchor가 약하고, 실험 구조가 여러 연구 축을 한 번에 포함해 병목을 분리하기 어렵다**는 점이다.

---

## 1. 진행 요약

```yaml
M0:
  status: closed
  summary:
    - repo scaffold
    - WSL/Linux LeWM training boundary
    - local artifact policy

M1:
  status: closed_with_caveats
  summary:
    - dataset source audit
    - open-schematics / SchGen / CircuitVQA / CGHD 역할 정리
    - license and provenance 기록

M2:
  status: closed
  summary:
    - CircuitSample schema
    - train/holdout/test curriculum
    - open-schematics usable pair and CGHD local pair manifest

M3:
  status: closed
  summary:
    - Qwen3-VL baseline 측정
    - smaller/quantized Qwen baseline은 명시적 defer

M4:
  status: closed
  summary:
    - LeWM data pipeline sanity
    - global view / tile view / structure target traceability 확인

M5:
  status: first_run_closed_with_caveats
  summary:
    - 5k open-schematics LeWM-S 학습 성공
    - loss 감소와 VRAM fit 확인
    - final holdout retrieval은 random top1을 넘지 못함

M5.3:
  status: closed_with_caveats
  summary:
    - ROI-aware graph/set target diagnostic 수행
    - train retrieval은 약하게 random 이상
    - final holdout top1은 random과 같음
```

---

## 2. 확인된 결과

긍정적으로 확인된 점:

- RTX 3090 / WSL 환경에서 LeWM-S 계열 학습을 실행할 수 있다.
- 5k open-schematics 학습과 512/128 ROI graph diagnostic 모두 VRAM 24GB 안에 들어왔다.
- 데이터셋 출처, split, raw artifact policy, result brief 형식은 어느 정도 정리됐다.
- 회로 도메인에서 component multi-hot target보다 KiCad graph/set + ROI trace target이 더 나은 방향이라는 점은 확인했다.

부정적 또는 불충분한 점:

- M5 첫 5k run의 final holdout top1 retrieval은 random과 같았다.
- M5.3 ROI graph diagnostic도 final holdout top1이 random과 같았다.
- train retrieval은 약하게 움직였지만, 안정적인 holdout evidence retention이라고 보기 어렵다.
- ROI/tile probe는 random baseline 자체가 높아 강한 증거로 쓰기 어렵다.
- 현재 구조는 LeWM evidence retention, offline teacher loop, ROI routing, taxonomy routing, LoRA/adapter bank를 동시에 건드리고 있어 실패 원인을 분리하기 어렵다.

---

## 3. 핵심 병목 1: Structure Anchor 정렬 문제

현재 병목은 단순히 “데이터가 부족하다”기보다 **정렬 anchor가 약하다**는 쪽에 가깝다.

```text
image latent
  -> 어떤 구조 표현을 기준으로 배워야 하는가?
```

M5 첫 run은 component multi-hot target을 사용했다. 이 target은 같은 component set을 가진 서로 다른 회로를 구분하기 어렵고, symbol 위치, wire, junction, net relation 같은 구조 정보를 충분히 담지 못했다.

M5.3에서는 KiCad parse 기반 ROI-aware graph/set target으로 바꿨지만, 여전히 완전한 topology target은 아니었다. 또한 image encoder와 structure encoder가 함께 움직이는 구조였기 때문에, loss는 줄어도 실제 회로 구조 의미를 안정적으로 보존한다는 증거가 약했다.

결과적으로 M5.3에서 `align_loss`는 크게 감소했지만 final holdout top1 retrieval은 random에 머물렀다. 이는 모델이 안정적인 구조 의미를 배웠다기보다, 움직이는 embedding 공간 안에서 batch-level objective를 어느 정도 맞춘 것일 가능성이 있다.

---

## 4. 핵심 병목 2: 회로 정적 이미지 도메인의 난도

회로 이미지는 LeWM 첫 성공 도메인으로는 난도가 높다.

회로 도메인에서 중요한 정보는 다음처럼 동시에 존재한다.

```yaml
hard_factors:
  - small text labels
  - thin wires
  - junction and crossing distinctions
  - symbol identity
  - component placement
  - net/topology relation
  - schematic-level global structure
```

224 resize나 단순 global view에서는 작은 문자와 선 정보가 쉽게 손실된다. Tile/ROI가 필요하지만, ROI routing 자체도 별도의 검증 주제가 된다.

또한 현재 회로 데이터는 정적 이미지 중심이다. LeWM/world-model의 장점은 시간적 변화, state transition, rollout consistency를 학습할 때 더 잘 드러날 수 있는데, 회로 정적 이미지에서는 이 장점이 잘 살아나지 않는다.

---

## 5. 핵심 병목 3: 실험 범위 과다

현재 프로젝트에는 다음 연구 축이 동시에 들어와 있다.

```yaml
research_axes:
  - compact Perception LeWM backbone
  - offline taxonomy / teacher loop
  - foveater-like ROI routing
  - multi-local LoRA or adapter bank
  - actual-only certification registry
  - Qwen3-VL monolithic comparison
```

이 축들은 각각 독립된 연구 주제가 될 만큼 크다. 한 레포 안에서 한 번에 검증하려고 하면 실패 원인이 perception, target, router, adapter, dataset 중 어디에 있는지 분리하기 어렵다.

특히 현재 결과에서는 backbone evidence retention 자체가 안정화되지 않았다. 이 상태에서 router, LoRA, ROI routing을 추가하면 성능이 나쁘게 나왔을 때 어느 모듈이 실패했는지 판단하기 어렵다.

---

## 6. 핵심 병목 4: Metric 해석 한계

현재 M5/M5.3에서 사용한 metric은 병목을 드러내는 데는 유용했지만, 강한 성공 증거로 쓰기에는 한계가 있었다.

```yaml
metric_limits:
  retrieval_top1:
    issue:
      - final holdout top1 stayed at random
      - small holdout에서는 epoch별 흔들림이 큼

  retrieval_top5:
    issue:
      - 일부 epoch에서 random보다 높았지만 안정적이지 않음

  tile_probe:
    issue:
      - positive tile 비율이 높아 random baseline 자체가 높음
      - weak improvement를 강한 ROI evidence로 해석하기 어려움

  train_loss:
    issue:
      - loss 감소가 structure evidence retention을 직접 보장하지 않음
```

따라서 현재 결과는 “학습이 실행되고 일부 weak signal이 있다”는 증거이지, “LeWM이 회로 구조 evidence를 충분히 보존한다”는 증거는 아니다.

---

## 7. 잠정 중단 판단

본 프로젝트는 현재 형태로 계속 확장하기보다 잠정 중단하는 편이 타당하다.

중단 사유:

- 핵심 실험 질문이 너무 넓다.
- 회로 정적 이미지 도메인은 LeWM 첫 성공 사례로 과하게 어렵다.
- M5/M5.3 결과는 partial signal은 있지만, 다음 단계 router/LoRA 실험으로 넘어갈 만큼 안정적이지 않다.
- LeWM backbone의 evidence retention이 먼저 안정화되어야 한다.

이 중단은 실패 폐기가 아니라 **병목을 확인한 뒤 범위를 줄이기 위한 중단**이다.

---

## 8. 보존할 산출물

보존할 가치가 있는 산출물:

- dataset source audit와 provenance 기록
- CircuitSample schema와 split manifest
- Qwen3 baseline result
- LeWM data pipeline sanity check
- M5 first run negative result
- M5.3 ROI graph diagnostic negative/partial result
- offline teacher loop protocol draft
- local artifact policy와 WSL training boundary

main claim으로 쓰지 않을 산출물:

- M5/M5.3 retrieval partial signal
- ROI tile probe 결과
- Gemma 4 offline teacher loop 초안
- routing/adapter/certification 관련 계획 문서

---

## 9. 현재 결론

```text
이 레포는 회로 도메인 compact LeWM + routing 구조를 탐색했지만,
M5/M5.3에서 LeWM evidence retention이 충분히 안정화되지 않았다.
주요 병목은 데이터 크기만이 아니라 structure anchor 정렬, 회로 정적 이미지 도메인의 난도,
그리고 backbone / ROI / routing / LoRA를 동시에 검증하려 한 범위 과다에 있다.
따라서 현재 구조를 확장하기보다 잠정 중단하고,
핵심 병목을 분리한 뒤 별도 방향에서 재설계하는 편이 안전하다.
```
