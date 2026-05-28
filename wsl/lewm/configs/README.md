# WSL LeWM Runtime Configs

Linux runtime configs for LeWM training go here.

These configs may reference Linux paths, but they must not embed private secrets or machine-specific absolute paths unless clearly marked as examples.

Current configs:

```text
circuit_lewm_s_5k.yaml
  - M5 first 5k open-schematics Circuit LeWM-S run
  - no sample-count inflation augmentation
  - local-only checkpoint policy

circuit_lewm_s_m5_3_roi_graph_512_128.yaml
  - M5.3 512/128 ROI-aware graph/set diagnostic
  - step-matched 30 epoch small-data run
  - local-only checkpoint policy
```
