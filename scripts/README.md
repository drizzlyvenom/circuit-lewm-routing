# Scripts

Root scripts are for cross-platform orchestration and evaluation.

Planned scripts:

```text
audit_sources.py
prepare_circuit_samples.py
run_lewm_data_pipeline_sanity.py
run_lewm_probe.py
train_router_llm.py
train_router_jepa.py
run_system_comparison.py
```

Do not place LeWM training entrypoints here. WSL/Linux world-model training entrypoints live under `wsl/lewm/scripts/`.

Current scripts:

```text
audit_sources.py
  - queries Hugging Face Hub API and Dataset Viewer API
  - writes data/circuit_sources/source_manifest.json
  - records license, source URLs, split metadata, parquet footprint, field previews, file inventory, and M1 decisions
  - does not download raw datasets or store signed asset URLs

prepare_circuit_samples.py
  - writes data/circuit_curricula/train.jsonl, holdout.jsonl, test.jsonl, and usable_pair_summary.json
  - verifies open-schematics image+schematic rows before adding them to the curriculum
  - pairs local CGHD image/XML files from ignored data/downloads
  - stores CircuitVQA row references without committing prompt/answer payloads

run_qwen_baseline.py
  - runs M3 Qwen3-VL single-backbone evaluation on deterministic CircuitVQA test QA refs
  - writes results/qwen/qwen3_single_backbone.json
  - can explicitly defer the smaller/quantized Qwen baseline into results/qwen/qwen_small_or_quantized.json
  - stores match flags, latency, and resource metrics without committing raw prompts, expected answers, or predictions

run_lewm_data_pipeline_sanity.py
  - runs M4 LeWM data pipeline sanity on 64 actual train records from data/circuit_curricula/train.jsonl
  - builds global image views, tile/crop views, structure target hashes/counts, and traceable tile metadata
  - writes results/lewm_data_pipeline/sanity_check.json
  - does not store raw images, raw schematics, raw XML, or tile pixels in committed results

audit_roi_structure_targets.py
  - audits M5 target redesign after the first LeWM-S 5k run failed to beat random top1 retrieval
  - parses open-schematics KiCad schematic text into graph/set and ROI trace statistics
  - audits local CGHD XML boxes as ROI/detail specialist supervision
  - writes results/structure_targets/roi_structure_target_audit.json
  - does not store raw images, raw schematics, raw XML, or raw label text
```
