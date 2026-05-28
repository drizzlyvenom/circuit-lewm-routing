# Scripts

Root scripts are for cross-platform orchestration and evaluation.

Planned scripts:

```text
audit_sources.py
prepare_circuit_samples.py
run_qwen_baseline.py
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
```
