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
  - records source URLs, license metadata, split/row counts, feature previews, and M1 decisions

prepare_circuit_samples.py
  - builds M2 train/holdout/test CircuitSample manifests
  - writes data/circuit_curricula/train.jsonl, holdout.jsonl, test.jsonl, and split_summary.json
  - stores Hugging Face row/file references, not raw payloads or signed asset URLs

check_circuit_curricula.py
  - validates M2 JSONL schema requirements
  - checks QA answer_type refs, structure refs, CGHD image/XML pairing refs, unique sample ids, and no source-group leakage across splits

run_qwen_baseline.py
  - loads local Qwen3-VL-4B-Instruct from models/qwen
  - evaluates the selected CircuitVQA test subset from data/circuit_curricula/test.jsonl
  - writes aggregate M3 resource/accuracy evidence under results/qwen
```
