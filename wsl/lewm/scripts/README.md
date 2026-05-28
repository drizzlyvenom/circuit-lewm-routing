# WSL LeWM Scripts

Linux-only LeWM training and latent export wrappers go here.

Planned scripts:

```text
export_latents.py
```

Current scripts:

```text
train_circuit_lewm_s.py
  - runs M5 Circuit LeWM-S structure pretraining under WSL/Linux
  - uses the official LeWM checkout for ViT-tiny/SIGReg-compatible components
  - writes summary evidence to results/lewm_s/pretrain_log.json
  - keeps checkpoints and step metrics local to the WSL runtime
```

Root-level scripts should call these through WSL only when needed; they should not duplicate the training loop.
