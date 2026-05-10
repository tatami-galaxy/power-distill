"""
Run Power-SMC on GRPO's "all-wrong" buffer to measure recovery rate.

Reads the JSONL buffer produced by train_grpo.py (prompts where all G rollouts
scored 0) and attempts to recover correct solutions via power sampling.

One-shot mode:
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.eval.ps_all_wrong \
        --model Qwen/Qwen3-4B \
        --buffer results/rl/all_wrong_buffer.jsonl \
        --alpha 4 --smc_particles 64

Watch mode (runs alongside GRPO, polls for new entries):
    CUDA_VISIBLE_DEVICES=4 uv run python -m src.eval.ps_all_wrong \
        --model Qwen/Qwen3-4B \
        --buffer results/rl/all_wrong_buffer.jsonl \
        --recovered results/rl/recovered_buffer.jsonl \
        --watch --watch_interval 30
"""

import argparse
import json
import os
import time

from tqdm import tqdm
from scalable_power_sampling import HFPowerSMCSampler
from src.utils import extract_boxed_answer, is_equiv
from src.eval.run_eval import build_prompt


def _get_problem_text(entry):
    if entry["prompt"] is not None:
        for msg in entry["prompt"]:
            if msg["role"] == "user":
                return msg["content"]
    return ""


def load_buffer(path: str, seen_keys: set | None = None) -> list[dict]:
    """Load all-wrong buffer JSONL. Returns only new (unseen) entries."""
    if seen_keys is None:
        seen_keys = set()
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            problem_text = _get_problem_text(entry)
            key = (problem_text, entry["answer"])
            if key not in seen_keys:
                seen_keys.add(key)
                entries.append(entry)
    return entries


def create_sampler(
    model_name, alpha, n_particles, ess_threshold, block_size,
    alpha_ramp_tokens, max_tokens, min_new_tokens, top_k, top_p,
    repetition_penalty, dtype, stop_on_boxed, use_cow_cache,
    shared_prompt_cache, chat_template_model,
):
    temperature = 1.0 / alpha
    sampler = HFPowerSMCSampler(
        model_name=model_name,
        alpha=alpha,
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        proposal_temperature=temperature,
        block_size=block_size,
        alpha_ramp_tokens=alpha_ramp_tokens,
        max_new_tokens=max_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        dtype=dtype,
        stop_on_boxed=stop_on_boxed,
        use_cow_cache=use_cow_cache,
        shared_prompt_cache=shared_prompt_cache,
    )
    template_tok = sampler.tokenizer
    if chat_template_model:
        from transformers import AutoTokenizer
        template_tok = AutoTokenizer.from_pretrained(
            chat_template_model, trust_remote_code=True
        )
    return sampler, template_tok


def recover_one(sampler, template_tok, entry, prompt_mode, enable_thinking):
    """Power-sample one entry. Returns result dict with 'correct' flag."""
    tokenizer = sampler.tokenizer
    problem_text = _get_problem_text(entry)
    gold = entry["answer"]

    prompt_str = build_prompt(
        problem_text, prompt_mode, tokenizer, template_tok, enable_thinking,
    )
    input_ids = tokenizer.encode(prompt_str)

    t0 = time.time()
    out = sampler.generate(input_ids=input_ids, verbose=False)
    elapsed = time.time() - t0

    response = out["text"]
    pred_answer = extract_boxed_answer(response)
    correct = is_equiv(pred_answer, gold) if pred_answer else False

    return {
        "prompt": entry["prompt"],
        "answer": gold,
        "completions": entry.get("completions"),  # original wrong rollouts
        "power_response": response,
        "pred_answer": pred_answer,
        "correct": correct,
        "num_tokens_generated": out["num_tokens_generated"],
        "sample_time_s": elapsed,
        "smc_stats": out.get("stats", {}),
        "grpo_step": entry.get("step"),
    }


def run_power_sampling(sampler, template_tok, entries, prompt_mode,
                       enable_thinking, recovered_path=None):
    """Run power sampling on a batch of entries. Streams recovered results to disk."""
    results = []
    recovered = 0

    pbar = tqdm(entries, desc="power_smc_recovery", unit="problem")
    for i, entry in enumerate(pbar):
        result = recover_one(sampler, template_tok, entry, prompt_mode, enable_thinking)
        results.append(result)

        if result["correct"]:
            recovered += 1
            # Stream recovered entries to disk immediately
            if recovered_path and result["completions"] is not None:
                with open(recovered_path, "a") as f:
                    f.write(json.dumps({
                        "prompt": result["prompt"],
                        "answer": result["answer"],
                        "completions": result["completions"],
                        "power_response": result["power_response"],
                    }) + "\n")

        pbar.set_postfix(
            recovered=f"{recovered}/{i+1}",
            rate=f"{100*recovered/(i+1):.1f}%",
            tokens=result["num_tokens_generated"],
            time=f"{result['sample_time_s']:.1f}s",
        )

    if entries:
        print(f"[Recovery] {recovered}/{len(entries)} "
              f"({100*recovered/len(entries):.1f}%) recovered")

    return results, recovered


def run_watch_mode(sampler, template_tok, buffer_path, recovered_path,
                   prompt_mode, enable_thinking, watch_interval):
    """Continuously poll buffer for new all-wrong entries and power-sample them."""
    seen_keys = set()
    total_processed = 0
    total_recovered = 0

    print(f"[Watch] Polling {buffer_path} every {watch_interval}s")
    print(f"[Watch] Writing recovered entries to {recovered_path}")

    os.makedirs(os.path.dirname(recovered_path) or ".", exist_ok=True)

    while True:
        new_entries = load_buffer(buffer_path, seen_keys)
        if new_entries:
            print(f"\n[Watch] Found {len(new_entries)} new all-wrong entries")
            _, recovered = run_power_sampling(
                sampler, template_tok, new_entries, prompt_mode,
                enable_thinking, recovered_path,
            )
            total_processed += len(new_entries)
            total_recovered += recovered
            rate = 100 * total_recovered / total_processed if total_processed else 0
            print(f"[Watch] Cumulative: {total_recovered}/{total_processed} "
                  f"({rate:.1f}%) recovered")
        else:
            print(f"[Watch] No new entries, sleeping {watch_interval}s...")

        time.sleep(watch_interval)


def save_results(results, recovered_count, output_dir, model_name, config):
    os.makedirs(output_dir, exist_ok=True)

    total = len(results)
    results_path = os.path.join(output_dir, "recovery_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "model": model_name,
        "total": total,
        "recovered": recovered_count,
        "recovery_rate": recovered_count / total if total else 0,
        "power_smc_config": config,
    }
    summary_path = os.path.join(output_dir, "recovery_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Power-SMC on GRPO all-wrong buffer to measure recovery rate"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--buffer", type=str, required=True,
                        help="Path to all_wrong_buffer.jsonl from train_grpo.py")
    parser.add_argument("--recovered", type=str, default=None,
                        help="Path to write recovered_buffer.jsonl (for Phase 3). "
                             "Defaults to <buffer_dir>/recovered_buffer.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/recovery")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--prompt_mode", type=str, default="chat",
                        choices=["chat", "raw"])
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Only evaluate first N entries from buffer (one-shot mode)")
    parser.add_argument("--chat_template_model", type=str, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"])

    # Watch mode
    parser.add_argument("--watch", action="store_true",
                        help="Continuously poll buffer for new entries (run alongside GRPO)")
    parser.add_argument("--watch_interval", type=int, default=30,
                        help="Seconds between polls in watch mode")

    # Power-SMC args
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--smc_particles", type=int, default=64)
    parser.add_argument("--smc_ess_threshold", type=float, default=0.5)
    parser.add_argument("--smc_block_size", type=int, default=64)
    parser.add_argument("--smc_alpha_ramp_tokens", type=int, default=400)
    parser.add_argument("--smc_min_new_tokens", type=int, default=100)
    parser.add_argument("--smc_top_k", type=int, default=0)
    parser.add_argument("--smc_top_p", type=float, default=0.9)
    parser.add_argument("--smc_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_smc_stop_on_boxed", action="store_true")
    parser.add_argument("--no_smc_cow_cache", action="store_true")
    parser.add_argument("--no_smc_shared_prompt_cache", action="store_true")

    args = parser.parse_args()

    # Default recovered path next to buffer
    if args.recovered is None:
        args.recovered = os.path.join(
            os.path.dirname(args.buffer), "recovered_buffer.jsonl"
        )

    # Create sampler once
    sampler, template_tok = create_sampler(
        model_name=args.model,
        alpha=args.alpha,
        n_particles=args.smc_particles,
        ess_threshold=args.smc_ess_threshold,
        block_size=args.smc_block_size,
        alpha_ramp_tokens=args.smc_alpha_ramp_tokens,
        max_tokens=args.max_tokens,
        min_new_tokens=args.smc_min_new_tokens,
        top_k=args.smc_top_k,
        top_p=args.smc_top_p,
        repetition_penalty=args.smc_repetition_penalty,
        dtype=args.dtype,
        stop_on_boxed=not args.no_smc_stop_on_boxed,
        use_cow_cache=not args.no_smc_cow_cache,
        shared_prompt_cache=not args.no_smc_shared_prompt_cache,
        chat_template_model=args.chat_template_model,
    )

    if args.watch:
        run_watch_mode(
            sampler, template_tok, args.buffer, args.recovered,
            args.prompt_mode, args.enable_thinking, args.watch_interval,
        )
    else:
        # One-shot mode
        entries = load_buffer(args.buffer)
        if not entries:
            print("Buffer is empty or does not exist yet.")
            return
        if args.num_samples is not None:
            entries = entries[:args.num_samples]
            print(f"  Using first {len(entries)} entries")

        results, recovered = run_power_sampling(
            sampler, template_tok, entries, args.prompt_mode,
            args.enable_thinking, args.recovered,
        )

        # Save full results
        config = {
            "alpha": args.alpha, "n_particles": args.smc_particles,
            "ess_threshold": args.smc_ess_threshold,
            "block_size": args.smc_block_size,
            "alpha_ramp_tokens": args.smc_alpha_ramp_tokens,
        }
        model_slug = args.model.replace("/", "_")
        save_results(results, recovered, os.path.join(args.output_dir, model_slug),
                     args.model, config)


if __name__ == "__main__":
    main()
