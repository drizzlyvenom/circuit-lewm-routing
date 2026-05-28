# M5 LeWM-S Structure Pretraining Result

Status: first 5k run closed_with_caveats
Measured at: 2026-05-29 KST

---

## what_was_run

M5 첫 실행에서는 `open-schematics` 5,000개 train image/structure pair로 `Circuit LeWM-S` 구조 alignment 학습을 수행했다.

```yaml
script: wsl/lewm/scripts/train_circuit_lewm_s.py
config_reference: wsl/lewm/configs/circuit_lewm_s_5k.yaml
summary_output: results/lewm_s/pretrain_log.json
runtime: WSL Ubuntu-24.04
official_repo: external_official_lewm_checkout
checkpoint_policy: local_only
```

실행은 WSL official LeWM checkout에서 수행했고, checkpoint와 step metrics CSV는 WSL runtime local-only 경로에 둔다. 레포에는 summary JSON과 결과 문서만 남긴다.

---

## dataset_and_split

```yaml
source_dataset: bshada/open-schematics
source_url: https://huggingface.co/datasets/bshada/open-schematics
license: cc-by-4.0
train_manifest: data/circuit_curricula/train.jsonl
holdout_manifest: data/circuit_curricula/holdout.jsonl
train_records: 5000
holdout_records: 512
component_vocab_size: 4096
augmentation_policy: none
```

이번 실행은 5k unique train samples를 사용했다. 표본 수를 늘리기 위한 augmentation은 적용하지 않았다. 단, masked-crop objective를 위해 sample당 deterministic crop view 하나를 만들었다.

---

## metrics

```yaml
status: closed_with_caveats
epochs_requested: 3
epochs_completed: 3
batch_size: 64
samples_seen_per_epoch: 4992
total_parameters: 16211904
trainable_parameters: 16211904
precision: bf16
peak_vram_mb_torch_allocated: 4785.837
peak_vram_mb_torch_reserved: 5310.0
process_ram_mb_after_train: 5334.012
elapsed_seconds: 235.931
checkpoint_size: 186M
```

최종 epoch 기준 retrieval은 다음과 같다.

```yaml
final_epoch:
  train_loss: 3.760675
  align_loss: 3.100455
  mask_loss: 0.005504
  struct_loss: 0.649476
  sigreg_loss: 3.665064
  holdout_alignment_retrieval_top1: 0.001953
  holdout_random_top1: 0.001953
  holdout_alignment_retrieval_top5: 0.011719
  holdout_random_top5: 0.009766
```

---

## result_table

| check | result | note |
|---|---:|---|
| training stable | true | 3 epochs completed without non-finite loss |
| loss decreased | true | loss `5.206547 -> 3.760675` |
| peak VRAM within 24GB | true | torch reserved peak `5310.0 MB` |
| local-only checkpoint saved | true | WSL runtime checkpoint, not committed |
| final holdout retrieval above random | false | top1 equals random, top5 slightly above random |

Epoch retrieval trace:

| epoch | loss | top1 | random top1 | top5 | random top5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.206547 | 0.003906 | 0.001953 | 0.011719 | 0.009766 |
| 2 | 4.296871 | 0.005859 | 0.001953 | 0.009766 | 0.009766 |
| 3 | 3.760675 | 0.001953 | 0.001953 | 0.011719 | 0.009766 |

---

## failure_modes

- 최종 checkpoint의 holdout top1 retrieval이 random과 같아서 M5 통과 조건을 완전히 만족하지 못했다.
- loss는 안정적으로 감소했지만, 이것만으로 image/structure alignment가 살아났다고 볼 수 없다.
- 현재 structure encoder는 component multi-hot 기반이라 schematic topology, net relation, symbol position 정보를 충분히 반영하지 못한다.
- component vocab coverage는 train mention 기준 `0.8643`, holdout mention 기준 `0.8184`라 OOV 구조 정보가 남아 있다.
- HF Hub 비인증 경고가 출력됐다. 실행은 성공했지만 반복 학습 전에는 cache/token 전략을 정리하는 편이 안전하다.

---

## claim_boundary

이번 M5 첫 run으로 말할 수 있는 것은 다음까지다.

- RTX 3090 / WSL 환경에서 5k `open-schematics` 기반 `Circuit LeWM-S` 학습은 메모리와 VRAM 안에 들어왔다.
- 16.21M parameter 모델은 3 epoch 동안 non-finite loss 없이 학습됐다.
- loss 감소와 checkpoint 저장은 확인됐다.
- 단, 최종 holdout retrieval이 random top1을 넘지 못했으므로, 회로 image/structure alignment가 충분하다는 주장은 아직 할 수 없다.

아직 말할 수 없는 것은 다음과 같다.

- LeWM-S latent가 회로 구조 evidence를 안정적으로 보존한다는 주장.
- 이 checkpoint를 Qwen3 baseline과 task accuracy로 비교할 수 있다는 주장.
- 5k open-schematics만으로 M5 통과가 가능하다는 주장.

---

## next_action

다음 M5 설계 수정은 학습량을 무작정 늘리기보다 structure target 품질을 먼저 손보는 쪽이 맞다.

```yaml
recommended_redesign:
  - replace pure component multi-hot target with schematic text/token or netlist-derived structure embedding
  - save best retrieval checkpoint instead of only latest
  - lower structure BCE dominance or rebalance component labels
  - add train-vs-holdout retrieval diagnostics before longer runs
  - keep unique sample count at 5k until alignment metric moves clearly above random
```
