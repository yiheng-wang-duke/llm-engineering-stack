"""
GSM8K & MATH-500 evaluation: generate reasoning + answer, then check exact match.

Supports multi-GPU parallel inference (same pattern as eval_leetcode.py).
Loads datasets from HuggingFace Hub by default, or from local JSONL files.

Usage:
    # GSM8K evaluation
    python eval_gsm8k_math500.py --model_path Qwen/Qwen3-4B --benchmark gsm8k

    # MATH-500 evaluation
    python eval_gsm8k_math500.py --model_path Qwen/Qwen3-4B --benchmark math500

    # multi-GPU
    python eval_gsm8k_math500.py --model_path Qwen/Qwen3-4B --benchmark gsm8k --num_gpus 4
"""

import argparse
import json
import os
import re
import sys

# allow imports from project root (parent of eval/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tqdm import tqdm
import multiprocessing

from model import load_model


# ── answer extraction ─────────────────────────────────────────────

def extract_answer(response: str) -> str:
    """Extract the final numerical answer from model response.

    Tries multiple patterns:
    1. \\boxed{...}
    2. "The answer is ..."
    3. "#### <number>"  (GSM8K gold format)
    4. Last number in the response
    """
    # 1. \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return _normalize_answer(boxed[-1])

    # 2. "The answer is ..."
    match = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)", response)
    if match:
        return _normalize_answer(match.group(1).strip())

    # 3. GSM8K gold format "#### <number>"
    match = re.search(r"####\s*(.+)", response)
    if match:
        return _normalize_answer(match.group(1).strip())

    # 4. Last number in the response
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", response)
    if numbers:
        return _normalize_answer(numbers[-1])

    return response.strip()


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip()
    # remove trailing period
    answer = answer.rstrip(".")
    # remove commas from numbers
    answer = answer.replace(",", "")
    # remove dollar signs, percent signs
    answer = answer.replace("$", "").replace("%", "")
    # remove leading/trailing whitespace again
    answer = answer.strip()
    return answer


def extract_gold_answer(sample: dict, benchmark: str) -> str:
    """Extract gold answer from a dataset sample."""
    if benchmark == "gsm8k":
        # GSM8K: answer is after "####" in the "answer" field
        answer_text = sample["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return _normalize_answer(match.group(1).strip())
        return _normalize_answer(answer_text.strip())
    else:
        # MATH-500: answer is in "answer" field, often in \boxed{}
        answer_text = sample.get("answer", sample.get("solution", ""))
        boxed = re.findall(r"\\boxed\{([^}]+)\}", answer_text)
        if boxed:
            return _normalize_answer(boxed[-1])
        return _normalize_answer(answer_text.strip())


def is_correct(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    if predicted == gold:
        return True
    # try numeric comparison
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        pass
    return False


# ── data loading ──────────────────────────────────────────────────

def load_benchmark(benchmark: str, data_path: str | None = None) -> list[dict]:
    """Load benchmark data from HuggingFace Hub or local path."""
    if data_path and os.path.exists(data_path):
        # local JSONL file
        from data_utils.read_jsonl import read_jsonl
        return read_jsonl(data_path)

    from datasets import load_dataset

    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        samples = []
        for i, item in enumerate(ds):
            samples.append({
                "id": f"gsm8k_{i}",
                "question": item["question"],
                "answer": item["answer"],
            })
        return samples
    elif benchmark == "math500":
        ds = load_dataset("HuggingFaceTB/MATH-500", split="test")
        samples = []
        for i, item in enumerate(ds):
            samples.append({
                "id": f"math500_{i}",
                "question": item["problem"],
                "answer": item.get("answer", item.get("solution", "")),
            })
        return samples
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# ── generation ────────────────────────────────────────────────────

def build_prompt(question: str) -> str:
    """Build the evaluation prompt for a math problem."""
    return (
        f"Solve the following math problem step by step. "
        f"Put your final answer within \\boxed{{}}.\n\n"
        f"{question}"
    )


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 2048) -> str:
    """Generate a single answer for a math question."""
    prompt = build_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response


# ── incremental save ──────────────────────────────────────────────

def _save_raw_result(output_dir: str, task_result: dict):
    """Append one task's raw result as a JSONL line."""
    path = os.path.join(output_dir, "raw_outputs.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(task_result, ensure_ascii=False) + "\n")


# ── single-problem evaluation ─────────────────────────────────────

def evaluate_problem(model, tokenizer, sample: dict, benchmark: str,
                     n_samples: int, max_new_tokens: int, output_dir: str) -> dict:
    """Generate n_samples answers for one problem, check correctness, save raw outputs."""
    question = sample["question"]
    gold = extract_gold_answer(sample, benchmark)
    task_id = sample.get("id", sample.get("task_id", "unknown"))

    samples = []
    n_correct = 0
    for i in range(n_samples):
        response = generate_answer(model, tokenizer, question, max_new_tokens)
        predicted = extract_answer(response)
        correct = is_correct(predicted, gold)
        if correct:
            n_correct += 1
        samples.append({
            "sample_idx": i,
            "raw_response": response,
            "predicted_answer": predicted,
            "correct": correct,
        })

    task_result = {
        "task_id": task_id,
        "question": question,
        "gold_answer": gold,
        "n_samples": n_samples,
        "n_correct": n_correct,
        "samples": samples,
    }
    _save_raw_result(output_dir, task_result)
    return task_result


# ── GPU worker (for multi-GPU) ────────────────────────────────────

def _gpu_worker(gpu_id: int, model_path: str, problems: list[dict],
                benchmark: str, n_samples: int, max_new_tokens: int,
                output_dir: str, result_queue: multiprocessing.Queue,
                worker_idx: int, total_workers: int):
    """Worker process: load model on a specific GPU, evaluate assigned problems."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model, tokenizer = load_model(model_path, use_device_map=False)
    model = model.to("cuda:0")
    model.eval()

    desc = f"GPU {gpu_id} ({len(problems)} problems)"
    for sample in tqdm(problems, desc=desc, position=worker_idx):
        result = evaluate_problem(model, tokenizer, sample, benchmark,
                                  n_samples, max_new_tokens, output_dir)
        result_queue.put(result)

    result_queue.put(None)


# ── main ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["gsm8k", "math500"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Local JSONL data path (overrides HuggingFace download)")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples per problem (for maj@k)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser.parse_args()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    if args.output is None:
        args.output = f"outputs/eval/{args.benchmark}/eval_results.json"

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_outputs.jsonl")
    if os.path.exists(raw_path):
        os.remove(raw_path)

    # load benchmark data
    test_data = load_benchmark(args.benchmark, args.data_path)
    print(f"Loaded {len(test_data)} problems from {args.benchmark}")

    results = []

    n_available = torch.cuda.device_count()
    n_gpus = min(args.num_gpus, n_available) if n_available > 0 else 0

    if n_gpus > 1:
        # ── multi-GPU path ────────────────────────────────────────
        chunks = [[] for _ in range(n_gpus)]
        for i, sample in enumerate(test_data):
            chunks[i % n_gpus].append(sample)

        result_queue = multiprocessing.Queue()
        workers = []
        for idx in range(n_gpus):
            p = multiprocessing.Process(
                target=_gpu_worker,
                args=(idx, args.model_path, chunks[idx],
                      args.benchmark, args.n_samples, args.max_new_tokens,
                      output_dir, result_queue, idx, n_gpus),
            )
            p.start()
            workers.append(p)

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

        for sample in tqdm(test_data, desc=f"Evaluating {args.benchmark}"):
            result = evaluate_problem(model, tokenizer, sample, args.benchmark,
                                      args.n_samples, args.max_new_tokens, output_dir)
            results.append(result)

    # compute accuracy
    total = len(results)
    correct = sum(1 for r in results if r["n_correct"] > 0)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{args.benchmark} Accuracy: {accuracy:.4f} ({correct}/{total})")

    # if n_samples > 1, also compute maj@k (majority voting)
    if args.n_samples > 1:
        maj_correct = 0
        for r in results:
            # majority vote across samples
            from collections import Counter
            predictions = [s["predicted_answer"] for s in r["samples"]]
            most_common = Counter(predictions).most_common(1)[0][0]
            if is_correct(most_common, r["gold_answer"]):
                maj_correct += 1
        maj_accuracy = maj_correct / total if total > 0 else 0.0
        print(f"{args.benchmark} Maj@{args.n_samples}: {maj_accuracy:.4f} ({maj_correct}/{total})")
    else:
        maj_accuracy = None

    # save summary
    summary = {
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
    }
    if maj_accuracy is not None:
        summary[f"maj@{args.n_samples}"] = maj_accuracy

    summary["details"] = [
        {
            "task_id": r["task_id"],
            "gold_answer": r["gold_answer"],
            "n_correct": r["n_correct"],
            "predicted_answer": r["samples"][0]["predicted_answer"],
        }
        for r in results
    ]

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {args.output}")
    print(f"Raw outputs saved to {raw_path}")


if __name__ == "__main__":
    main()