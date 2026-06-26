"""
LeetCodeDataset evaluation: generate code completions and run test cases (pass@k).

Supports multi-GPU parallel inference — each GPU loads an independent model copy
and processes a subset of problems. Raw outputs (all n_samples per problem) are
saved incrementally after each problem finishes.

Usage:
    # single GPU (default)
    python eval.py --model_path checkpoints/epoch_1 --data_path ../data/coding/LeetCodeDataset

    # multi-GPU (use 4 GPUs)
    python eval.py --model_path Qwen/Qwen3-4B --data_path ../data/coding/LeetCodeDataset --num_gpus 4 --k 1 5
"""

import argparse
import glob
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

# allow imports from project root (parent of eval/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tqdm import tqdm

from model import load_model
from data_utils.read_jsonl import read_jsonl


# ── execution sandbox ──────────────────────────────────────────────

def _run_test(code: str, timeout: int = 30) -> bool:
    """Execute generated code + test in an isolated subprocess with timeout.

    Uses subprocess instead of multiprocessing to avoid re-importing heavy
    modules (torch, transformers, etc.) which would exceed the timeout.
    """
    import subprocess
    import tempfile

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tmp.write(code)
        tmp.close()
        result = subprocess.run(
            [sys.executable, tmp.name],
            timeout=timeout,
            capture_output=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(tmp.name)


# ── generation ─────────────────────────────────────────────────────

def generate_completion(model, tokenizer, query: str, max_new_tokens: int = 1024) -> str:
    """Generate a single completion given a query prompt."""
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response


def extract_code(response: str) -> str:
    """Extract python code from model response (handles ```python blocks)."""
    if "```python" in response:
        blocks = response.split("```python")
        if len(blocks) > 1:
            code = blocks[1].split("```")[0]
            return code.strip()
    if "```" in response:
        blocks = response.split("```")
        if len(blocks) > 1:
            code = blocks[1].split("```")[0]
            return code.strip()
    return response.strip()


# ── pass@k metric ─────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k given n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c, n - c - k, -1)) / math.prod(range(n, n - k, -1))


# ── incremental save ──────────────────────────────────────────────

def _save_raw_result(output_dir: str, task_result: dict):
    """Append one task's raw result as a JSONL line (thread/process-safe via append mode)."""
    path = os.path.join(output_dir, "raw_outputs.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(task_result, ensure_ascii=False) + "\n")


# ── single-problem evaluation ─────────────────────────────────────

def evaluate_problem(model, tokenizer, sample: dict, n_samples: int,
                     max_new_tokens: int, output_dir: str) -> dict:
    """Generate n_samples completions for one problem, test them, and save raw outputs."""
    task_id = sample["task_id"]
    prompt_code = sample["prompt"]
    test_code = sample["test"]
    entry_point = sample["entry_point"]
    query = sample["query"]

    samples = []
    n_correct = 0
    for i in range(n_samples):
        response = generate_completion(model, tokenizer, query, max_new_tokens)
        code = extract_code(response)
        full_code = f"{prompt_code}\n{code}\n{test_code}\ncheck({entry_point})\n"
        passed = _run_test(full_code)
        if passed:
            n_correct += 1
        samples.append({
            "sample_idx": i,
            "raw_response": response,
            "extracted_code": code,
            "passed": passed,
        })

    task_result = {
        "task_id": task_id,
        "query": query,
        "n_samples": n_samples,
        "n_correct": n_correct,
        "samples": samples,
    }
    # incremental save — write immediately after finishing this problem
    _save_raw_result(output_dir, task_result)
    return task_result


# ── GPU worker (for multi-GPU) ────────────────────────────────────

def _gpu_worker(gpu_id: int, model_path: str, problems: list[dict],
                n_samples: int, max_new_tokens: int, output_dir: str,
                result_queue: multiprocessing.Queue, worker_idx: int,
                total_workers: int):
    """Worker process: load model on a specific GPU, evaluate assigned problems."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # After restricting visibility, the single GPU appears as cuda:0
    model, tokenizer = load_model(model_path, use_device_map=False)
    model = model.to("cuda:0")
    model.eval()

    desc = f"GPU {gpu_id} ({len(problems)} problems)"
    for sample in tqdm(problems, desc=desc, position=worker_idx):
        result = evaluate_problem(model, tokenizer, sample, n_samples,
                                  max_new_tokens, output_dir)
        result_queue.put(result)

    # signal this worker is done
    result_queue.put(None)


# ── main ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=5,
                        help="number of samples per problem")
    parser.add_argument("--k", type=int, nargs="+", default=[1],
                        help="k values for pass@k")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for parallel inference. "
                             "Automatically selects the first N available GPUs.")
    return parser.parse_args()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    # prepare output directory
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_outputs.jsonl")
    # clear previous raw outputs if any
    if os.path.exists(raw_path):
        os.remove(raw_path)

    # load test data
    test_files = glob.glob(f"{args.data_path}/*test*.jsonl")
    test_data = read_jsonl(test_files[0])
    print(f"Loaded {len(test_data)} test problems")

    results = []

    # detect available GPUs
    n_available = torch.cuda.device_count()
    n_gpus = min(args.num_gpus, n_available) if n_available > 0 else 0

    if n_gpus > 1:
        # ── multi-GPU path ────────────────────────────────────────
        gpu_ids = list(range(n_gpus))
        # split problems roughly evenly across GPUs
        chunks = [[] for _ in range(n_gpus)]
        for i, sample in enumerate(test_data):
            chunks[i % n_gpus].append(sample)

        result_queue = multiprocessing.Queue()
        workers = []
        for idx, gpu_id in enumerate(gpu_ids):
            p = multiprocessing.Process(
                target=_gpu_worker,
                args=(gpu_id, args.model_path, chunks[idx],
                      args.n_samples, args.max_new_tokens, output_dir,
                      result_queue, idx, n_gpus),
            )
            p.start()
            workers.append(p)

        # collect results from all workers
        done_count = 0
        pbar = tqdm(total=len(test_data), desc="Total progress", position=n_gpus)
        while done_count < n_gpus:
            item = result_queue.get()
            if item is None:
                done_count += 1
            else:
                results.append(item)
                pbar.update(1)
        pbar.close()

        for p in workers:
            p.join()

    else:
        # ── single-GPU path ──────────────────────────────────────

        model, tokenizer = load_model(args.model_path, use_device_map=True)
        model.eval()

        for sample in tqdm(test_data, desc="Evaluating"):
            result = evaluate_problem(model, tokenizer, sample, args.n_samples,
                                      args.max_new_tokens, output_dir)
            results.append(result)

    # compute pass@k
    total = len(results)
    pass_at_k_scores = {}
    for k in args.k:
        scores = [pass_at_k(r["n_samples"], r["n_correct"], k) for r in results]
        avg = sum(scores) / total
        pass_at_k_scores[str(k)] = avg
        print(f"pass@{k}: {avg:.4f} ({total} problems)")

    # save summary results
    summary = {
        "pass_at_k": pass_at_k_scores,
        "details": [
            {"task_id": r["task_id"], "n_samples": r["n_samples"], "n_correct": r["n_correct"]}
            for r in results
        ],
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.output}")
    print(f"Raw outputs saved to {raw_path}")


if __name__ == "__main__":
    main()
