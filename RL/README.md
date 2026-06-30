# GRPO RLVR Pipeline (math + code)

A minimal **RL-with-Verifiable-Rewards** pipeline built on top of
[verl](https://github.com/volcengine/verl). It trains a model with **GRPO** on a
mix of **math** (GSM8K) problems. Rewards are *verifiable*:

| ability | dataset | `data_source` | verifier |
|---------|---------|---------------|----------|
| math    | GSM8K   | `openai/gsm8k`| exact-match on the `#### answer` (verl `gsm8k` reward) |


verl already implements GRPO and both reward functions; this repo only adds the
 **launch config** that wire them together. No model code is modified.

## Prerequisites
- A working **verl** install (its own conda/venv with `ray`, `vllm`, `torch`,
  `flash-attn`, etc.). 
  