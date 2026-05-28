# M5.3 ROI Graph Diagnostic Result

Status: closed_with_caveats
Measured at: 2026-05-29 KST

---

## what_was_run

M5.3에서는 M5 첫 5k run의 `component multi-hot` target을 버리고, KiCad schematic에서 파싱한 ROI-aware graph/set target으로 `Circuit LeWM-S` diagnostic 학습을 수행했다.

이번 실행은 큰 학습으로 바로 돌아가지 않고 `512/128` split에서 target과 objective가 실제로 신호를 내는지 확인하는 목적이다. 512개 train에 대해 30 epoch를 사용해, 첫 M5 5k run의 3 epoch와 비슷한 update step 규모로 맞췄다.

```yaml
script: wsl/lewm/scripts/train_circuit_lewm_s_roi_graph.py
config_reference: wsl/lewm/configs/circuit_lewm_s_m5_3_roi_graph_512_128.yaml
summary_output: results/lewm_s/m5_3_roi_graph_diagnostic.json
runtime: WSL Ubuntu-24.04
official_repo: external_official_lewm_checkout
checkpoint_policy: local_only
```

---

## dataset_and_split

```yaml
source_dataset: bshada/open-schematics
source_url: https://huggingface.co/datasets/bshada/open-schematics
license: cc-by-4.0
train_manifest: data/circuit_curricula/train.jsonl
holdout_manifest: data/circuit_curricula/holdout.jsonl
train_records: 512
holdout_records: 128
augmentation_policy: none
crop_policy: deterministic ROI crop from parsed symbol positions
```

출력에는 raw image, raw schematic text, raw label text, checkpoint를 저장하지 않는다.

---

## metrics

```yaml
status: closed_with_caveats
epochs_requested: 30
epochs_completed: 30
batch_size: 64
samples_seen_per_epoch: 512
total_parameters: 11959706
trainable_parameters: 11959706
precision: bf16
peak_vram_mb_torch_allocated: 4725.03
peak_vram_mb_torch_reserved: 5248.0
process_ram_mb_after_train: 3666.039
elapsed_seconds: 92.259
```

Target 구성:

```yaml
target_mode: roi_graph_set
target_dim: 602
lib_vocab_size: 512
family_vocab_size: 64
tile_4x4_size: 16
target_fields:
  - symbol_lib_count_vector
  - symbol_family_count_vector
  - tile_4x4_symbol_occupancy
  - symbol/wire/label/junction/count scalars
  - bbox/layout scalars
```

최종 epoch 기준:

```yaml
final_epoch: 30
train_alignment_retrieval_top1: 0.005859
train_random_top1: 0.001953
train_alignment_retrieval_top5: 0.017578
train_random_top5: 0.009766
holdout_alignment_retrieval_top1: 0.007812
holdout_random_top1: 0.007812
holdout_alignment_retrieval_top5: 0.046875
holdout_random_top5: 0.039062
holdout_tile_topk_recall: 0.702617
holdout_tile_random_topk_recall: 0.704365
holdout_target_mse: 0.054183
```

Best holdout retrieval epoch:

```yaml
epoch: 5
holdout_alignment_retrieval_top1: 0.015625
holdout_random_top1: 0.007812
holdout_alignment_retrieval_top5: 0.054688
holdout_random_top5: 0.039062
holdout_tile_topk_recall: 0.736188
holdout_tile_random_topk_recall: 0.704365
```

---

## result_table

| check | result | note |
|---|---:|---|
| training stable | true | 30 epochs completed without non-finite loss |
| loss decreased | true | `6.154455 -> 0.961205` |
| train retrieval above random | true | final train top1/top5 both above random, but weak |
| final holdout top1 above random | false | final top1 equals random |
| final holdout top5 above random | true | `0.046875` vs random `0.039062` |
| best holdout top1 above random | true | epoch 5 top1 `0.015625` vs random `0.007812` |
| final tile probe above random | false | final tile top-k recall slightly below random |
| best tile probe above random | true | epoch 5 and epoch 22 were above random |
| peak VRAM within 24GB | true | torch reserved peak `5248.0 MB` |
| local-only checkpoint saved | true | WSL runtime checkpoint, not committed |

Epoch trace subset:

| epoch | loss | top1 | random top1 | top5 | random top5 | tile recall | tile random |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.154455 | 0.007812 | 0.007812 | 0.031250 | 0.039062 | 0.724499 | 0.704365 |
| 5 | 4.136104 | 0.015625 | 0.007812 | 0.054688 | 0.039062 | 0.736188 | 0.704365 |
| 10 | 2.746442 | 0.007812 | 0.007812 | 0.039062 | 0.039062 | 0.721302 | 0.704365 |
| 20 | 1.518217 | 0.007812 | 0.007812 | 0.046875 | 0.039062 | 0.701835 | 0.704365 |
| 30 | 0.961205 | 0.007812 | 0.007812 | 0.046875 | 0.039062 | 0.702617 | 0.704365 |

---

## failure_modes

- 학습 loss와 target MSE는 개선됐지만, final holdout top1 retrieval은 random과 같다.
- train retrieval은 random보다 높지만 매우 약하다. 따라서 모델이 구조 target을 강하게 외웠다고 보기도 어렵고, 안정적으로 일반화했다고 보기도 어렵다.
- holdout top1/top5와 tile probe가 epoch 5에서 잠깐 random을 넘었지만, 최종 epoch까지 안정적으로 유지되지 않았다.
- ROI-aware graph/set target은 `component multi-hot`보다 낫지만, 현재 contrastive objective와 ViT-tiny from-scratch 조합만으로는 robust evidence retention을 만들기에 부족하다.
- 4x4 tile occupancy는 positive tile 비율이 높아서 random baseline 자체가 높다. tile metric은 보조 지표로만 해석해야 한다.
- HF Hub 비인증 경고가 출력됐다. 실행은 성공했지만 반복 실행 전에는 cache/token 전략을 정리하는 편이 안전하다.

---

## claim_boundary

이번 M5.3으로 말할 수 있는 것은 다음까지다.

- `512/128` ROI-aware graph/set diagnostic은 RTX 3090 / WSL 환경에서 안정적으로 실행됐다.
- Target 차원을 `602`로 줄인 LeWM-S variant는 약 `11.96M` trainable parameter와 약 `5.25GB` torch reserved VRAM으로 학습 가능했다.
- `component multi-hot`보다 구조적인 target을 넣으면 아주 약한 train retrieval 신호와 일시적인 holdout 신호는 관찰된다.
- 그러나 이 결과만으로 LeWM-S latent가 회로 구조 evidence를 안정적으로 보존한다고 말할 수 없다.

아직 말할 수 없는 것은 다음과 같다.

- M5를 pass로 닫아도 된다는 주장.
- 이 LeWM checkpoint를 M6/M7 이후 router 실험의 안정적인 backbone으로 써도 된다는 주장.
- 데이터셋을 늘리면 같은 objective가 자연스럽게 해결된다는 주장.
- ROI detail을 LoRA/전문가가 맡기면 현재 backbone 문제가 자동으로 보정된다는 주장.

---

## next_action

M5를 닫는다면 `passed`가 아니라 `closed_with_caveats` 또는 `valid_negative_partial_signal`에 가깝다.

계속 밀어붙일 경우 다음은 데이터 크기 확장보다 objective ablation이 먼저다.

```yaml
recommended_next_if_continue:
  - save best checkpoint by holdout retrieval instead of only latest
  - compare contrastive retrieval vs direct supervised structure/tile probe
  - freeze or precompute the structure target projection to reduce moving-target instability
  - separate global structure target from ROI/tile target losses
  - report train retrieval, holdout retrieval, and tile probe together for every M5 follow-up
  - design Gemma 4 offline teacher anchor audit before scaling pseudo-labels
```

초기 LoRA ROI-detail 설계는 아직 깨지지 않았다. 다만 지금 확인된 병목은 LoRA가 맡을 디테일 문제가 아니라, LeWM backbone이 global image에서 안정적인 구조 latent를 만드는 단계에 있다.
