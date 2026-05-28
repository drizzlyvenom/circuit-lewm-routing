# Circuit LeWM 학습 참조 문서

Status: training reference draft
Scope: Perception LeWM objectives, model tiers, and first-run policy

---

## 0. 목표

Perception LeWM은 회로 이미지를 바로 답변으로 변환하는 모델이 아니라, 회로 구조 evidence를 latent에 보존하는 모델이다.

첫 학습 목표:

```text
schematic image
  -> circuit structure latent
  -> component / symbol / relation evidence
```

---

## 1. Objective

### 1.1 Image-to-Structure Contrastive Alignment

이미지 latent \(z_I\)와 structure latent \(z_S\)를 가깝게 한다.

\[
\mathcal{L}_{align}
=
-\log
\frac{\exp(\mathrm{sim}(z_I,z_S)/\tau)}
{\sum_{S'}\exp(\mathrm{sim}(z_I,z_{S'})/\tau)}
\]

목표:

```yaml
goal:
  - rendered schematic image와 KiCad/JSON/YAML structure가 같은 회로를 가리키도록 정렬
```

### 1.2 Masked Crop / Neighbor Prediction

부분 schematic crop을 보고 연결된 주변 crop 또는 held-out symbol latent를 예측한다.

\[
\mathcal{L}_{mask}
=
\left\|\hat{z}_{crop}-\mathrm{sg}(z_{crop})\right\|_2^2
\]

Pseudo-transition 후보:

```yaml
pseudo_transitions:
  - full_schematic_to_crop
  - crop_to_neighbor_crop
  - symbol_view_to_component_metadata
  - partial_schematic_to_missing_connection
  - augmented_render_to_same_structure_latent
```

### 1.3 Structure Prediction

회로 구조를 직접 예측하는 auxiliary head를 둔다.

```yaml
structure_heads:
  - component_presence
  - component_type
  - symbol_count
  - connection_or_net_edge
  - pin_or_net_relation_if_extractable
```

\[
\mathcal{L}_{struct}
=
\lambda_c \mathcal{L}_{component}
+
\lambda_e \mathcal{L}_{edge}
+
\lambda_n \mathcal{L}_{net}
\]

### 1.4 Total Loss

\[
\mathcal{L}_{total}
=
\lambda_a \mathcal{L}_{align}
+
\lambda_m \mathcal{L}_{mask}
+
\lambda_s \mathcal{L}_{struct}
\]

초기값:

```yaml
loss_weights_initial:
  lambda_align: 1.0
  lambda_mask: 1.0
  lambda_struct: 0.5
```

---

## 2. Model Tiers

### LeWM-S

```yaml
name: lewm_s
role: first actual run
encoder: ViT-tiny
image_size: 224 or 336
patch_size: 14
embed_dim: 192
predictor_depth: 6
estimated_trainable_parameters: about 18M
recommendation: primary_start
```

### LeWM-S+

```yaml
name: lewm_s_plus
role: underfit diagnosis
encoder: ViT-tiny
embed_dim: 192
predictor_depth: 8
estimated_trainable_parameters: about 22M
```

### LeWM-M-small

```yaml
name: lewm_m_small
role: paper candidate if LeWM-S underfits
encoder: ViT-small
embed_dim: 384
predictor_depth: 4
estimated_trainable_parameters: about 38M
```

### LeWM-M-plus

```yaml
name: lewm_m_plus
role: upper local candidate
encoder: ViT-small
embed_dim: 384
predictor_depth: 6
estimated_trainable_parameters: about 45M
```

### Deferred

```yaml
defer:
  - ViT-base / LeWM-XL
  - full latent predictor adaptation
  - full vision encoder LoRA
```

---

## 3. First-Run Policy

처음부터 256k unique image를 요구하지 않는다. `open-schematics`의 실제 usable image/structure pair를 audit하고, 부족하면 render augmentation과 crop/pseudo-transition으로 늘린다.

```yaml
first_run:
  model: LeWM-S
  data:
    - open-schematics image/structure pairs
  view:
    - global 224 or 336
    - tile/crop probe after pipeline is stable
  objectives:
    - image_to_structure_alignment
    - masked_crop_or_neighbor_prediction
  validation:
    - holdout retrieval
    - component/symbol probes
```

확장 조건:

```yaml
scale_to_lewm_m_if:
  - LeWM-S probe above random
  - LeWM-S underfits end-to-end task
  - VRAM headroom remains safe

do_not_scale_if:
  - data pipeline is unstable
  - LeWM-S latent collapses
  - probe metrics are random
```
