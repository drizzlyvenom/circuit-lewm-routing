# M4 LeWM Data Pipeline Result

Status: M4 closed
Measured at: 2026-05-29 KST

---

## what_was_run

M4에서는 M2 `CircuitSample` train curriculum에서 LeWM 학습용 image/structure batch를 실제로 구성했다.

```yaml
script: scripts/run_lewm_data_pipeline_sanity.py
output: results/lewm_data_pipeline/sanity_check.json
train_manifest: data/circuit_curricula/train.jsonl
batch:
  open_schematics: 32
  cghd: 32
  total_records: 64
global_resolution: 224
tile_resolution: 224
tiles_per_sample: 4
```

이 실행은 LeWM 학습이 아니다. Windows 루트에서는 dataloader sanity와 manifest/metadata 검증까지만 수행하고, 실제 LeWM 학습은 M5에서 WSL/Linux 영역으로 넘긴다.

---

## dataset_and_split

```yaml
split: train
sources:
  bshada/open-schematics:
    selected_records: 32
    role: image_to_kicad_schematic_structure_pair
    license: cc-by-4.0
  lowercaseonly/cghd:
    selected_records: 32
    role: image_to_symbol_annotation_pair
    license: cc-by-3.0
raw_payload_policy:
  stores_raw_images: false
  stores_raw_schematics: false
  stores_raw_xml: false
  stores_tile_pixels: false
  stores_hashes_and_shapes_only: true
```

`open-schematics`는 HF row reference를 따라 같은 row의 image와 schematic text를 함께 읽고, `CGHD`는 ignored local download 아래 image/XML pair를 함께 읽었다.

---

## metrics

```yaml
status: closed
selected_records: 64
global_views_shape: [64, 3, 224, 224]
tile_views_shape: [64, 4, 3, 224, 224]
tile_metadata_count: 256
target_types:
  kicad_schematic_text: 32
  cghd_xml_symbol_annotations: 32
resources:
  elapsed_seconds: 25.643
  rss_mb_before: 546.113
  rss_mb_after: 1844.246
  rss_mb_delta: 1298.133
  ram_limit_mb: 26624
```

---

## result_table

| check | result | note |
|---|---:|---|
| one batch loaded | true | 64 records |
| RAM under 24~26GB policy | true | RSS after load 1844.246 MB |
| global view shape | true | `[64, 3, 224, 224]` |
| tile view shape | true | `[64, 4, 3, 224, 224]` |
| image/structure alignment | true | sample_id와 structure target hash가 함께 검증됨 |
| tile/crop traceability | true | 256 tile metadata records |
| mixed structure sources | true | open-schematics + CGHD |

---

## failure_modes

- HF Hub 비인증 요청 경고가 출력됐다. 실행은 성공했지만 대량 반복 실행 전에는 HF token 또는 local cache 전략을 정리하는 편이 안전하다.
- 이번 실행은 첫 64개 deterministic train records를 사용한 batch sanity다. 64개는 학습 규모로는 너무 작으며, 전체 15,208개 train curriculum 전체를 materialize한 결과도 아니다.
- `open-schematics`의 KiCad schematic text는 hash/length/component count로만 결과에 남겼다. 원문 schematic을 커밋하지 않으므로, 구조 세부 디버깅은 HF 원본 row를 다시 조회해야 한다.
- augmentation은 structure-preserving 수준의 deterministic photometric augmentation만 확인했다. 회전, 강한 crop, elastic transform 같은 aggressive augmentation은 아직 적용하지 않았다.

---

## claim_boundary

이번 M4의 active evidence로 말할 수 있는 것은 다음까지다.

- LeWM 학습에 들어갈 64개 image/structure batch가 실제로 로딩됐다.
- global view와 tile view tensor shape가 LeWM-S 첫 실험에 사용할 수 있는 형태로 만들어졌다.
- 각 image는 structure target hash 또는 XML hash와 sample_id 단위로 정렬됐다.
- 모든 tile/crop metadata는 원본 sample, 원본 image ref, 원본 좌표계 crop box로 되돌아갈 수 있다.
- 한 batch 로딩은 RAM 24~26GB 정책보다 충분히 낮은 RSS에서 끝났다.

아직 말할 수 없는 것은 다음과 같다.

- LeWM-S 학습이 안정적으로 수렴한다는 주장.
- 이 pipeline만으로 latent가 회로 구조 evidence를 보존한다는 주장.
- 224 해상도와 4개 tile이 최적이라는 주장.
- 전체 train curriculum materialization이 HF rate limit 없이 항상 성공한다는 주장.
- 64개 batch sanity 결과가 M5 학습 규모나 최종 성능을 대표한다는 주장.

---

## next_action

M5에서는 이 pipeline을 WSL/Linux LeWM runtime으로 넘기고, `LeWM-S` structure pretraining을 실제로 수행한다. 첫 M5 run은 `open-schematics` structure pair를 중심으로 두고, `CGHD`는 perception probe 또는 crop/ROI sanity 보조 evidence로 사용한다.
