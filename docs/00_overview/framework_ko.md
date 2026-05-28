# Circuit LeWM Routing 프레임워크

Status: framework draft
Scope: compact circuit perception/world-model + certified routing architecture

---

## 0. 요약

이 프레임워크의 목표는 단일 Qwen3-VL backbone이 perception, reasoning, routing, answering을 모두 맡는 구조를 작은 world-model perception 구조로 분해해 검증하는 것이다.

```text
Qwen3 monolithic VLM
  vs
Perception LeWM
  -> router
  -> answer head / certified adapter
```

핵심 claim은 `LoRA alone saves VRAM`이 아니다. 핵심은 **회로 구조 evidence를 작은 latent에 보존하는 compact perception/world-model backbone이 성립하는가**다.

---

## 1. 시스템 분해

### 1.1 Perception LeWM

Perception LeWM은 회로 이미지를 곧바로 답으로 바꾸는 모델이 아니라, 다음 evidence를 latent로 보존하는 모델이다.

```yaml
inputs:
  - schematic_image
  - optional_tile_or_crop
  - optional_render_augmentation
  - optional_structure_text

outputs:
  - latent_state
  - visual_evidence_summary
  - component_relation_features
  - uncertainty_or_confidence
```

### 1.2 Router

Router는 LeWM이 만든 latent/evidence와 query를 보고 taxonomy 또는 adapter id를 고른다.

```yaml
router_variants:
  llm_router:
    inputs:
      - user_query
      - visual_evidence_summary
      - component_relation_summary
      - AdapterCard registry summary
    outputs:
      - predicted_taxonomy
      - selected_adapter_id
      - abstain_flag
      - route_confidence

  jepa_latent_router:
    inputs:
      - latent_state
      - query_embedding
      - adapter_registry_embedding
    outputs:
      - predicted_taxonomy
      - selected_adapter_id
      - abstain_flag
      - route_confidence
```

### 1.3 Answer / Adapter Head

LeWM 구조에서 adapter slot은 language decoder에 고정되지 않는다.

```yaml
adapter_slots:
  structure_projector:
    recommended: true
  taxonomy_router_head:
    recommended: true
  answer_head:
    recommended: true

defer_initially:
  - full perception_encoder adaptation
  - latent_predictor adaptation
```

---

## 2. Global + Tile Pipeline

회로 이미지는 작은 문자, 핀, 선, component label이 중요하다. Full schematic을 224로 줄이면 evidence가 사라질 수 있으므로 global view와 tile view를 분리한다.

```yaml
visual_pipeline:
  global_view:
    resolution: 224 or 336
    role:
      - coarse layout
      - global circuit topology

  tile_views:
    resolution: 224 or 336 per tile
    role:
      - symbol evidence
      - text label
      - wire/connection detail

  aggregation:
    method:
      - attention pooling
      - graph pooling
      - set transformer
```

초기 LeWM-S도 global-only 결과를 main claim으로 바로 해석하지 않는다. 최소한 tile/crop probe를 통해 evidence 손실 여부를 확인한다.

---

## 3. 비교군

```yaml
baselines:
  B0_qwen3_vl_single:
    role: strong_monolithic_baseline

  B1_smaller_or_quantized_qwen:
    role: practical_low_vram_monolithic_baseline

experimental:
  E1_perception_lewm_lora_routing_llm:
    role: compact_perception_with_llm_router

  E2_perception_lewm_jepa_router:
    role: compact_perception_with_latent_router

ablation:
  E3_perception_lewm_simple_head:
    role: latent_information_probe_before_routing_claim
```

---

## 4. 검증 순서

```text
dataset source audit
  -> Qwen baseline
  -> LeWM data pipeline
  -> LeWM-S pretraining
  -> frozen latent probe
  -> router baseline
  -> answer / adapter head eval
  -> actual-only certification
  -> end-to-end comparison
```

Router/LoRA 결과는 perception evidence retention gate를 통과하기 전에는 main claim으로 해석하지 않는다.
