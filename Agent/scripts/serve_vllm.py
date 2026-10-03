#!/usr/bin/env python3
"""Launch in a separate environment with vLLM installed; read .env without shell evaluation."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["chat", "embedding"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print arguments with API key redacted"
    )
    args, extra = parser.parse_known_args()
    env = os.environ.copy()
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        from dotenv import dotenv_values

        for key, value in dotenv_values(env_file).items():
            if value is not None:
                env.setdefault(key, value)
    chat = args.kind == "chat"
    prefix = "AW_LLM" if chat else "AW_EMBEDDING"
    model = env.get(
        prefix + "_MODEL", "Qwen/Qwen3-4B-Instruct-2507" if chat else "Qwen/Qwen3-Embedding-0.6B"
    )
    executable = shutil.which("vllm") or "vllm"
    cmd = [
        executable,
        "serve",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port or (8000 if chat else 8001)),
        "--api-key",
        env.get(prefix + "_API_KEY", "local-dev-key"),
        "--max-model-len",
        env.get("VLLM_MAX_MODEL_LEN", "16384") if chat else "4096",
        "--gpu-memory-utilization",
        env.get("VLLM_CHAT_MEMORY" if chat else "VLLM_EMBED_MEMORY", "0.65" if chat else "0.20"),
        "--max-num-seqs",
        "8" if chat else "16",
    ]
    if chat:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    else:
        cmd += ["--runner", "pooling"]
    cmd += extra
    env["CUDA_VISIBLE_DEVICES"] = env.get("VLLM_CHAT_GPU" if chat else "VLLM_EMBED_GPU", "0")
    if args.dry_run:
        safe = list(cmd)
        safe[safe.index("--api-key") + 1] = "<redacted>"
        print("CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"])
        print(" ".join(safe))
        return
    if not shutil.which("vllm"):
        parser.error("vllm not found; activate the separate inference environment first")
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
