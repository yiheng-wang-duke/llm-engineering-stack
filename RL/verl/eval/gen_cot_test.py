"""
Generate CoT (Chain-of-Thought) reasoning for LeetCode samples
using claude-sonnet-4-6 via LiteLLM proxy.

The model is given the ground-truth solution and only generates the
<think>...</think> reasoning block. The final response is assembled as:
    <think>{generated_reasoning}</think>\n\n{original_response}

Saves results as JSONL with token usage and cost tracking.

Usage:
    python gen_cot_test.py
"""

import json
import os
import time

import requests

# --- Config ---
API_BASE = "http://hl279-cmp-01.egr.duke.edu:4000"
API_KEY = "sk-I5cqowry4TwTP_kYUAfgIw"
MODEL = "claude-sonnet-4-6"
DATA_PATH = "/home/yw784/Projects/SFT/data/coding/LeetCodeDataset/LeetCodeDataset-train.jsonl"
OUTPUT_PATH = "/home/yw784/Projects/SFT/sft_from_stratch/outputs/cot_test/claude_sonnet_4_6_cot_10.jsonl"
N_SAMPLES = 10

# Claude Sonnet 4.6 pricing (per million tokens)
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00


def read_jsonl(path, n=None):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
            if n and len(data) >= n:
                break
    return data


def call_api(session, query, ground_truth_response, max_tokens=2048):
    """Call LiteLLM proxy to generate ONLY the thinking/reasoning block.

    The model sees the problem AND the ground-truth solution, and is asked
    to produce the reasoning that leads to that solution.
    """
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

    user_prompt = (
        f"{query}\n\n"
        f"--- Correct Solution ---\n"
        f"{ground_truth_response}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    resp = session.post(f"{API_BASE}/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    text = data["choices"][0]["message"].get("content", "").strip()
    return text, usage


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load first N samples
    samples = read_jsonl(DATA_PATH, N_SAMPLES)
    print(f"Loaded {len(samples)} samples")

    # Set up session
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    results = []

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{N_SAMPLES}] task_id={sample['task_id']} difficulty={sample['difficulty']}")

        try:
            t0 = time.time()
            thinking, usage = call_api(session, sample["query"], sample["response"])
            elapsed = time.time() - t0

            # Assemble final response: <think>reasoning</think> + original answer
            assembled_response = f"<think>\n{thinking}\n</think>\n\n{sample['response']}"

            # Accumulate usage
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

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
                "response": assembled_response,  # <think>CoT</think> + original answer
                "thinking": thinking,  # raw thinking only
                "original_response": sample["response"],  # original non-CoT
                "model": MODEL,
                "usage": usage,
                "elapsed_seconds": round(elapsed, 2),
            }
            results.append(result)

            # Write incrementally
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            print(f"  done in {elapsed:.1f}s | prompt_tokens={pt} completion_tokens={ct}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # --- Summary ---
    pt = total_usage["prompt_tokens"]
    ct = total_usage["completion_tokens"]
    tt = total_usage["total_tokens"]
    cost_input = pt / 1_000_000 * PRICE_INPUT
    cost_output = ct / 1_000_000 * PRICE_OUTPUT
    cost_total = cost_input + cost_output

    summary = {
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "n_success": len(results),
        "total_prompt_tokens": pt,
        "total_completion_tokens": ct,
        "total_tokens": tt,
        "cost_input_usd": round(cost_input, 6),
        "cost_output_usd": round(cost_output, 6),
        "cost_total_usd": round(cost_total, 6),
        "price_per_sample_usd": round(cost_total / len(results), 6) if results else 0,
    }

    summary_path = OUTPUT_PATH.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Model: {MODEL}")
    print(f"Samples: {len(results)}/{N_SAMPLES}")
    print(f"Prompt tokens:     {pt:>10,}")
    print(f"Completion tokens: {ct:>10,}")
    print(f"Total tokens:      {tt:>10,}")
    print(f"Cost (input):   ${cost_input:.6f}")
    print(f"Cost (output):  ${cost_output:.6f}")
    print(f"Cost (total):   ${cost_total:.6f}")
    print(f"Cost per sample: ${cost_total / len(results):.6f}" if results else "")
    print(f"\nResults: {OUTPUT_PATH}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()