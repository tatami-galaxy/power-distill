"""
Run Power-SMC on GRPO's "all-wrong" buffer to measure recovery rate.

Reads the JSONL buffer produced by train_grpo.py (prompts where all G rollouts
scored 0) and attempts to recover correct solutions via power sampling.

Usage:

    CUDA_VISIBLE_DEVICES=0 uv run python -m src.eval.ps_all_wrong \
        --model Qwen/Qwen3-4B \
        --buffer results/rl/all_wrong_buffer.jsonl \
        --alpha 4 --smc_particles 64
"""

import argparse
import json
import os
import time

from tqdm import tqdm
from scalable_power_sampling import HFPowerSMCSampler
from src.utils import extract_boxed_answer, is_equiv
from src.eval.run_eval import build_prompt


def load_buffer(path: str) -> list[dict]:
    """Load all-wrong buffer JSONL. Each line has: prompt, answer, step."""
    entries = []
    seen = set()
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            # Deduplicate by answer + problem text
            problem_text = ""
            if entry["prompt"] is not None:
                for msg in entry["prompt"]:
                    if msg["role"] == "user":
                        problem_text = msg["content"]
                        break
            key = (problem_text, entry["answer"])
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    print(f"Loaded {len(entries)} unique all-wrong prompts from {path}")
    return entries


def run_power_sampling(
    model_name: str,
    entries: list[dict],
    alpha: float,
    n_particles: int,
    max_tokens: int,
    ess_threshold: float,
    block_size: int,
    alpha_ramp_tokens: int,
    min_new_tokens: int,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    dtype: str,
    stop_on_boxed: bool,
    use_cow_cache: bool,
    shared_prompt_cache: bool,
    chat_template_model: str | None,
    enable_thinking: bool | None,
    prompt_mode: str,
) -> dict:
    temperature = 1.0 / alpha

    print(f"\n{'='*60}")
    print(f"Power-SMC recovery on {len(entries)} all-wrong prompts")
    print(f"  model={model_name}")
    print(f"  alpha={alpha}, particles={n_particles}, temp={temperature:.4f}")
    print(f"  block_size={block_size}, alpha_ramp_tokens={alpha_ramp_tokens}")
    print(f"{'='*60}")

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
    tokenizer = sampler.tokenizer

    template_tok = tokenizer
    if chat_template_model:
        from transformers import AutoTokenizer
        template_tok = AutoTokenizer.from_pretrained(
            chat_template_model, trust_remote_code=True
        )

    results = []
    recovered = 0
    t0 = time.time()

    pbar = tqdm(entries, desc="power_smc_recovery", unit="problem")
    for i, entry in enumerate(pbar):
        # Extract problem text from chat prompt
        problem_text = ""
        if entry["prompt"] is not None:
            for msg in entry["prompt"]:
                if msg["role"] == "user":
                    problem_text = msg["content"]
                    break

        gold = entry["answer"]

        prompt_str = build_prompt(
            problem_text, prompt_mode, tokenizer, template_tok, enable_thinking,
        )
        input_ids = tokenizer.encode(prompt_str)

        sample_t0 = time.time()
        out = sampler.generate(input_ids=input_ids, verbose=False)
        sample_elapsed = time.time() - sample_t0

        response = out["text"]
        pred_answer = extract_boxed_answer(response)
        correct = is_equiv(pred_answer, gold) if pred_answer else False
        recovered += int(correct)

        stats = out.get("stats", {})
        results.append({
            "problem": problem_text,
            "answer": gold,
            "grpo_step": entry.get("step"),
            "response": response,
            "pred_answer": pred_answer,
            "correct": correct,
            "num_tokens_generated": out["num_tokens_generated"],
            "sample_time_s": sample_elapsed,
            "smc_stats": stats,
        })

        pbar.set_postfix(
            recovered=f"{recovered}/{i+1}",
            rate=f"{100*recovered/(i+1):.1f}%",
            tokens=out["num_tokens_generated"],
            time=f"{sample_elapsed:.1f}s",
        )

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Recovery results:")
    print(f"  Total all-wrong prompts: {len(entries)}")
    print(f"  Recovered by Power-SMC:  {recovered} ({100*recovered/len(entries):.1f}%)")
    print(f"  Still unsolved:          {len(entries)-recovered}")
    print(f"  Total time:              {elapsed:.1f}s")
    print(f"{'='*60}")

    return {
        "model": model_name,
        "total": len(entries),
        "recovered": recovered,
        "recovery_rate": recovered / len(entries) if entries else 0,
        "elapsed_s": elapsed,
        "power_smc_config": {
            "alpha": alpha,
            "n_particles": n_particles,
            "ess_threshold": ess_threshold,
            "temperature": temperature,
            "block_size": block_size,
            "alpha_ramp_tokens": alpha_ramp_tokens,
        },
        "results": results,
    }


def save_results(output: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Full results
    results_path = os.path.join(output_dir, "recovery_results.json")
    with open(results_path, "w") as f:
        json.dump(output["results"], f, indent=2)

    # Recovered samples only (for future SDFT)
    recovered_path = os.path.join(output_dir, "recovered_for_sdft.jsonl")
    with open(recovered_path, "w") as f:
        for r in output["results"]:
            if r["correct"]:
                f.write(json.dumps({
                    "problem": r["problem"],
                    "answer": r["answer"],
                    "response": r["response"],
                }) + "\n")

    # Summary
    summary = {k: v for k, v in output.items() if k != "results"}
    summary_path = os.path.join(output_dir, "recovery_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {recovered_path} ({output['recovered']} samples for SDFT)")
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Power-SMC on GRPO all-wrong buffer to measure recovery rate"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--buffer", type=str, required=True,
                        help="Path to all_wrong_buffer.jsonl from train_grpo.py")
    parser.add_argument("--output_dir", type=str, default="results/recovery")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--prompt_mode", type=str, default="chat",
                        choices=["chat", "raw"])
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Only evaluate first N entries from buffer")
    parser.add_argument("--chat_template_model", type=str, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"])

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

    # Load buffer
    entries = load_buffer(args.buffer)
    if args.num_samples is not None:
        entries = entries[:args.num_samples]
        print(f"  Using first {len(entries)} entries")

    # Run power sampling
    output = run_power_sampling(
        model_name=args.model,
        entries=entries,
        alpha=args.alpha,
        n_particles=args.smc_particles,
        max_tokens=args.max_tokens,
        ess_threshold=args.smc_ess_threshold,
        block_size=args.smc_block_size,
        alpha_ramp_tokens=args.smc_alpha_ramp_tokens,
        min_new_tokens=args.smc_min_new_tokens,
        top_k=args.smc_top_k,
        top_p=args.smc_top_p,
        repetition_penalty=args.smc_repetition_penalty,
        dtype=args.dtype,
        stop_on_boxed=not args.no_smc_stop_on_boxed,
        use_cow_cache=not args.no_smc_cow_cache,
        shared_prompt_cache=not args.no_smc_shared_prompt_cache,
        chat_template_model=args.chat_template_model,
        enable_thinking=args.enable_thinking,
        prompt_mode=args.prompt_mode,
    )

    # Save
    model_slug = args.model.replace("/", "_")
    save_results(output, os.path.join(args.output_dir, model_slug))


if __name__ == "__main__":
    main()
