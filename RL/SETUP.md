# Environment Setup

Create a fresh conda env and install verl + a recent transformers.

```bash
# 1. new conda env (Python 3.12 matches the prebuilt flash-attn wheel)
conda create -n verl python=3.12 -y
conda activate verl

# 2. vLLM 0.11.0 — the version verl pins for cu128/torch2.8.
#    The PyPI wheel is prebuilt (no source build) and pulls torch==2.8.0 (cu128) in.
pip install vllm==0.11.0

pip install verl==0.8.0

# 5. transformers > 4.57
pip install -U "transformers>=4.57"
```

> Why vllm 0.11.0: it's what verl's own installer pins for this stack
> (`scripts/install_vllm_sglang_mcore.sh`) and the cu128/torch2.8 docker image.
> Installing it first fixes torch at 2.8.0, so the bundled flash-attn wheel matches.

## Verify
```bash
python -c "import verl, transformers, vllm, flash_attn; \
print('verl', verl.__version__); print('transformers', transformers.__version__)"
```
`transformers` should print **≥ 4.57**.

Also check the rollout backend imports:
```bash
python -c "import vllm; print('vllm', vllm.__version__)"   # -> 0.11.0
```

## Notes
- Run these on a GPU node (the login node has no CUDA).
- vllm 0.11.0 installs a prebuilt cu128 wheel; if your driver is older than CUDA
  12.8, pick the matching verl docker tag instead of installing locally.
- If the bundled flash-attn wheel ever mismatches, fall back to
  `pip install flash-attn==2.7.4.post1 --no-build-isolation`.
- After setup, submit jobs from within the activated `verl` env
  (`conda activate verl && sbatch verl/train_grpo_math.sh`).
