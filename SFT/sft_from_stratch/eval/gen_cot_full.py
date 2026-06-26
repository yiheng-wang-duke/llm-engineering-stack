"""
Generate CoT (Chain-of-Thought) reasoning for ALL LeetCode training samples
using claude-sonnet-4-6 via LiteLLM proxy, with concurrent requests.

The model is given the ground-truth solution and only generates the
<think>...</think> reasoning block. The final response is assembled as:
    <think>{generated_reasoning}</think>\n\n{original_response}

Supports resume: skips already-completed task_ids found in the output file.

Usage:
    python eval/gen_cot_full.py
    python eval/gen_cot_full.py --workers 8    # more concurrency
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# --- Config ---
API_BASE = "http://hl279-cmp-01.egr.duke.edu:4000"
API_KEY = "sk-I5cqowry4TwTP_kYUAfgIw"
MODEL = "claude-sonnet-4-6"
DATA_PATH = "/home/yw784/Projects/SFT/data/coding/LeetCodeDataset/LeetCodeDataset-train.jsonl"
OUTPUT_PATH = "/home/yw784/Projects/SFT/data/coding/LeetCodeDataset/LeetCodeDataset-train-cot.jsonl"
SUMMARY_PATH = "/home/yw784/Projects/SFT/data/coding/LeetCodeDataset/LeetCodeDataset-train-cot-summary.json"

# Claude Sonnet 4.6 pricing (per million tokens)
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent API requests")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per sample")
    return parser.parse_args()


def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_completed(path):
    """Load already-completed task_ids for resume support."""
    completed = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        completed.add(r["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return completed


def call_api(session, query, ground_truth_response, max_retries=3):
    """Call LiteLLM proxy to generate ONLY the thinking/reasoning block."""
    system_prompt = (
        "You are an expert Python programmer. "
        "You will be given a coding problem and its correct solution. "
        "Your task is to generate ONLY the thinking/reasoning process that "
        "leads to this solution.\n\n"
        "Requirements:\n"
        "1. You MUST output ONLY the reasoning text — no code, no solution, no markdown.\n"
        "2. Do NOT wrap your output in <think> tags — just output the raw reasoning text.\n"
        "3. Your reasoning should include: understanding the problem, identifying key "
        "insights, considering edge cases, choosing the algorithm/data structure, "
        "and briefly analyzing time/space complexity.\n"
        "4. The reasoning should naturally lead to the given solution.\n"
        "5. Keep it concise but thorough (100-300 words)."
    )

    user_prompt = f"{query}\n\n--- Correct Solution ---\n{ground_truth_response}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 2048,
    }

    for attempt in range(max_retries):
        try:
            resp = session.post(f"{API_BASE}/chat/completions", json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            text = data["choices"][0]["message"].get("content", "").strip()
            return text, usage
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise


# Thread-safe file writer
_write_lock = threading.Lock()


def append_jsonl(path, obj):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_sample(session, sample, max_retries):
    """Process one sample: generate thinking, assemble response."""
    thinking, usage = call_api(session, sample["query"], sample["response"], max_retries)
    assembled_response = f"<think>\n{thinking}\n</think>\n\n{sample['response']}"

    result = {
        "task_id": sample["task_id"],
        "question_id": sample["question_id"],
        "difficulty": sample["difficulty"],
        "tags": sample["tags"],
        "problem_description": sample["problem_description"],
        "starter_code": sample["starter_code"],
        "prompt": sample["prompt"],
        "completion": sample["completion"],
        "entry_point": sample["entry_point"],
        "test": sample["test"],
        "input_output": sample["input_output"],
        "query": sample["query"],
        "response": assembled_response,
        "thinking": thinking,
        "original_response": sample["response"],
    }
    return result, usage


def main():
    args = parse_args()

    # Load data
    all_samples = read_jsonl(DATA_PATH)
    print(f"Total samples in dataset: {len(all_samples)}")

    # Resume support
    completed = load_completed(OUTPUT_PATH)
    remaining = [s for s in all_samples if s["task_id"] not in completed]
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All samples already completed. Nothing to do.")
    else:
        # Set up session
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        })

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _usage_lock = threading.Lock()
        n_success = 0
        n_error = 0

        def worker(sample):
            nonlocal n_success, n_error
            try:
                result, usage = process_sample(session, sample, args.max_retries)
                append_jsonl(OUTPUT_PATH, result)
                with _usage_lock:
                    for k in total_usage:
                        total_usage[k] += usage.get(k, 0)
                    n_success += 1
                return sample["task_id"], True
            except Exception as e:
                with _usage_lock:
                    n_error += 1
                return sample["task_id"], False

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, s): s for s in remaining}
            pbar = tqdm(total=len(remaining), desc="Generating CoT")
            for future in as_completed(futures):
                task_id, ok = future.result()
                if not ok:
                    tqdm.write(f"  FAILED: {task_id}")
                pbar.update(1)
            pbar.close()

        # Print usage summary
        pt = total_usage["prompt_tokens"]
        ct = total_usage["completion_tokens"]
        tt = total_usage["total_tokens"]
        cost_input = pt / 1_000_000 * PRICE_INPUT
        cost_output = ct / 1_000_000 * PRICE_OUTPUT
        cost_total = cost_input + cost_output

        print(f"\n{'=' * 60}")
        print(f"This run: {n_success} success, {n_error} failed")
        print(f"Prompt tokens:     {pt:>10,}")
        print(f"Completion tokens: {ct:>10,}")
        print(f"Total tokens:      {tt:>10,}")
        print(f"Cost (input):   ${cost_input:.4f}")
        print(f"Cost (output):  ${cost_output:.4f}")
        print(f"Cost (total):   ${cost_total:.4f}")
        if n_success > 0:
            print(f"Cost per sample: ${cost_total / n_success:.6f}")
        print(f"{'=' * 60}")

    # --- Final summary over ALL completed data ---
    all_results = read_jsonl(OUTPUT_PATH)
    # Deduplicate by task_id
    seen = set()
    unique = []
    for r in all_results:
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            unique.append(r)

    summary = {
        "model": MODEL,
        "total_in_dataset": len(all_samples),
        "total_completed": len(unique),
        "output_path": OUTPUT_PATH,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinal: {len(unique)}/{len(all_samples)} samples completed")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
