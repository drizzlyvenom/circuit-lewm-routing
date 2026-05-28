# Qwen vs LeWM 비교 프로토콜

Status: protocol draft
Scope: monolithic VLM baseline vs compact Perception LeWM systems

---

## 0. 목적

Qwen3-VL 단일 backbone과 Perception LeWM 기반 시스템을 같은 회로 task subset에서 비교한다.

이 비교는 단순히 Qwen3를 이기는지가 아니라 다음 tradeoff를 측정한다.

```text
accuracy / structure retention / resident VRAM / peak VRAM / latency / parameter count
```

---

## 1. 비교군

```yaml
systems:
  B0_qwen3_vl_single_backbone:
    role: strong_monolithic_baseline
    components:
      perception: Qwen3-VL
      reasoning: Qwen3-VL
      routing: none
      adapters: none

  B1_smaller_or_quantized_qwen:
    role: practical_low_vram_monolithic_baseline
    candidates:
      - Qwen2-VL-2B_or_3B_class
      - Qwen3-VL_quantized_if_available

  E1_perception_lewm_lora_routing_llm:
    role: experimental_group_a
    components:
      perception: Perception LeWM
      router: lightweight LLM router
      registry: certified AdapterCard registry

  E2_perception_lewm_jepa_router:
    role: experimental_group_b
    components:
      perception: Perception LeWM
      router: JEPA latent router
      registry: certified AdapterCard registry

  E3_perception_lewm_simple_head:
    role: ablation
    components:
      perception: frozen Perception LeWM
      head: linear_or_mlp_head
```

---

## 2. 데이터셋 역할

```yaml
structure_pretraining:
  open_schematics:
    role:
      - schematic image
      - KiCad / structured circuit representation
      - component metadata
      - image-to-structure alignment
      - optional offline teacher anchor labels if separately audited

  schgen_dataset:
    role:
      - circuit generation code/text prior
      - schematic grammar prior
      - curriculum / teacher prompt source

vqa_evaluation:
  CircuitVQA:
    role:
      - circuit image question answering
      - Qwen3 baseline comparison
      - end-to-end accuracy benchmark

perception_probe:
  CGHD:
    role:
      - handwritten circuit diagram symbol grounding
      - detection/segmentation-style probe
      - symbol evidence retention
```

Dataset source audit가 끝나기 전에는 score를 paper evidence로 쓰지 않는다.

---

## 3. 측정 항목

### Accuracy / Structure

```yaml
accuracy_metrics:
  - circuit_vqa_score
  - normalized_answer_match
  - structured_field_match
  - component_presence_f1
  - symbol_type_f1
  - structure_retrieval_top1
  - connection_edge_f1_or_proxy
```

### Resource

```yaml
resource_metrics:
  - trainable_parameters
  - total_parameters
  - checkpoint_size_mb
  - resident_vram_mb
  - peak_vram_mb
  - latency_ms
  - throughput_samples_per_second
```

### Router

```yaml
routing_metrics:
  - router_top1_hit
  - router_regret
  - abstain_rate
  - wrong_adapter_damage
```

---

## 4. Gate

### Gate P - Perception Evidence Retention

```yaml
pass_if:
  - structure_retrieval_top1 > random_baseline
  - at least two probe metrics are above random or trivial baseline
  - failure cases are attributable
```

Gate P 전에는 router나 answer head 결과를 main claim으로 해석하지 않는다.

### Gate R - Router Selection

```yaml
pass_if:
  - router_top1_hit > random_baseline
  - router_regret <= 0.05_to_0.10
  - abstain_rate is measured
  - wrong_adapter_damage is bounded
```

### Gate A - Answer / Task Score

```yaml
compare:
  - Qwen3 single backbone
  - smaller_or_quantized_Qwen
  - LeWM + LLM router
  - LeWM + JEPA router
```

성공 수준:

```yaml
level_1_perception_success:
  - structure retrieval above random
  - component/symbol probe meaningful
  - latent preserves circuit evidence

level_2_task_success:
  - LeWM system beats low-resource baseline
  - LeWM system approaches smaller Qwen baseline
  - failure modes are attributable

level_3_strong_success:
  - LeWM system within 5~10pp of Qwen3
  - peak_vram <= 50~60% of Qwen3
```

---

## 5. 실행 정책

Qwen baseline과 LeWM training은 같은 process에서 섞지 않는다.

```yaml
process_policy:
  - run Qwen baseline separately
  - run LeWM training separately
  - run router/answer head training separately
  - measure memory and latency in clean process
```

3090 실행 기준:

```yaml
hardware:
  gpu: RTX 3090
  usable_vram_gb: 24
  usable_ram_gb: 24_to_26

training_policy:
  precision:
    - fp16
    - bf16 if stable
  batch:
    - start small
    - use gradient accumulation
  dataloader:
    - cache preprocessed metadata
    - avoid loading huge full-resolution images into RAM all at once
```

---

## 6. 성공 기준

```yaml
strong_success:
  - LeWM system within 5pp to 10pp of Qwen3 on selected circuit tasks
  - peak_vram <= 0.50 to 0.60 of Qwen3
  - router wrong-adapter damage bounded
  - perception evidence probe clearly above random

moderate_success:
  - LeWM system beats smaller low-resource baseline
  - structure probes are strong
  - failure modes are attributable
  - resource usage is substantially lower than Qwen3

valid_negative:
  - LeWM accuracy much lower than Qwen3
  - but perception/router/answer failure is decomposed
  - resource metrics are measured
  - next model tier or objective change is clear
```
