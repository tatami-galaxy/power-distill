"""
Evaluate language models on MATH and other math benchmarks.

Usage:

    CUDA_VISIBLE_DEVICES=0 uv run python -m src.eval.run_eval \
    --model Qwen/Qwen2.5-Math-7B \
    --dataset math500 \
    --num_samples 10 \
    --power_smc \
    --alpha 4 \
    --smc_particles 64 \
    --smc_alpha_ramp_tokens 100
"""

import argparse
import json
import os
import time
from collections import defaultdict
from vllm import LLM, SamplingParams
from src.utils import (
    extract_boxed_answer,
    is_equiv,
    DATASET_REGISTRY_EVAL,
)
from scalable_power_sampling import HFPowerSMCSampler

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step.  "
    "Put your final answer in \\boxed{}."
)

RAW_PROMPT = "Can you solve the following math problem? "
RAW_COT = " Please reason step by step, and put your final answer within \\boxed{}."


def format_prompt_chat(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def format_prompt_raw(problem: str) -> str:
    return RAW_PROMPT + problem + RAW_COT


def build_prompt(problem: str, prompt_mode: str, tokenizer, template_tok=None,
                 enable_thinking=None) -> str:
    """Build a prompt string from a problem, respecting prompt_mode."""
    if prompt_mode == "raw":
        return format_prompt_raw(problem)
    messages = format_prompt_chat(problem)
    tok = template_tok or tokenizer
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tok.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------    

def evaluate_model(
    model_name: str,
    problems: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    chat_template_tokenizer=None,
    enable_thinking: bool | None = None,
    dtype: str = "bfloat16",
    prompt_mode: str = "chat",
) -> dict:
    """Run evaluation and return results dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Problems:   {len(problems)}")
    print(f"{'='*60}")

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype=dtype,
    )
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Build prompts
    template_tok = chat_template_tokenizer or tokenizer
    prompts = []
    for p in problems:
        prompt = build_prompt(
            p["problem"], prompt_mode, tokenizer, template_tok, enable_thinking,
        )
        prompts.append(prompt)

    # Generate
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - t0
    print(f"Generation took {elapsed:.1f}s ({len(problems)/elapsed:.1f} problems/s)")

    # Score
    results = []
    for prob, output in zip(problems, outputs):
        completion = output.outputs[0]
        response = completion.text
        pred_answer = extract_boxed_answer(response)
        correct = is_equiv(pred_answer, prob["answer"]) if pred_answer else False
        results.append({
            **prob,
            "response": response,
            "pred_answer": pred_answer,
            "correct": correct,
            "num_tokens_generated": len(completion.token_ids),
        })

    return {
        "model": model_name,
        "results": results,
        "elapsed_s": elapsed,
        "max_tokens": max_tokens,
    }


def evaluate_model_power_smc(
    model_name: str,
    problems: list[dict],
    max_tokens: int = 2048,
    alpha: float = 4.0,
    n_particles: int = 64,
    ess_threshold: float = 0.5,
    block_size: int = 64,
    alpha_ramp_tokens: int = 100,
    min_new_tokens: int = 0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    chat_template_tokenizer=None,
    enable_thinking: bool | None = None,
    dtype: str = "bfloat16",
    stop_on_boxed: bool = True,
    use_cow_cache: bool = True,
    shared_prompt_cache: bool = True,
    prompt_mode: str = "chat",
) -> dict:
    """Run evaluation using Power-SMC and return results dict."""
    from tqdm import tqdm

    method = "power_smc"
    temperature = 1.0 / alpha

    print(f"\n{'='*60}")
    print(f"Evaluating ({method}): {model_name}")
    print(
        f"  alpha={alpha}, particles={n_particles}, "
        f"ESS threshold={ess_threshold}"
    )
    print(
        f"  block_size={block_size}, alpha_ramp_tokens={alpha_ramp_tokens}, "
        f"temperature={temperature:.4f}"
    )
    print(f"Problems: {len(problems)}")
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

    results = []
    t0 = time.time()

    template_tok = chat_template_tokenizer or tokenizer

    correct_so_far = 0
    pbar = tqdm(problems, desc=method, unit="problem")
    for i, prob in enumerate(pbar):
        prompt_text = build_prompt(
            prob["problem"], prompt_mode, tokenizer, template_tok, enable_thinking,
        )
        input_ids = tokenizer.encode(prompt_text)

        sample_t0 = time.time()
        out = sampler.generate(input_ids=input_ids, verbose=False)
        sample_elapsed = time.time() - sample_t0

        response = out["text"]
        pred_answer = extract_boxed_answer(response)
        correct = is_equiv(pred_answer, prob["answer"]) if pred_answer else False
        correct_so_far += int(correct)

        stats = out.get("stats", {})
        results.append({
            **prob,
            "response": response,
            "pred_answer": pred_answer,
            "correct": correct,
            "num_tokens_generated": out["num_tokens_generated"],
            "sample_time_s": sample_elapsed,
            "smc_stats": stats,
        })

        pbar.set_postfix(
            acc=f"{correct_so_far}/{i+1}",
            tokens=out["num_tokens_generated"],
            resamples=stats.get("resample_count", 0),
            time=f"{sample_elapsed:.1f}s",
        )

    elapsed = time.time() - t0
    print(f"{method} took {elapsed:.1f}s total")

    config = {
        "alpha": alpha,
        "n_particles": n_particles,
        "ess_threshold": ess_threshold,
        "temperature": temperature,
        "block_size": block_size,
        "alpha_ramp_tokens": alpha_ramp_tokens,
        "min_new_tokens": min_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "stop_on_boxed": stop_on_boxed,
        "use_cow_cache": use_cow_cache,
        "shared_prompt_cache": shared_prompt_cache,
    }

    return {
        "model": model_name,
        "method": method,
        "power_smc_config": config,
        "results": results,
        "elapsed_s": elapsed,
        "max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(eval_output: dict):
    """Print accuracy breakdown by level and subject."""
    model = eval_output["model"]
    results = eval_output["results"]
    total = len(results)
    correct = sum(r["correct"] for r in results)

    print(f"\n{'='*60}")
    print(f"Results: {model}")
    print(f"Overall: {correct}/{total} = {correct/total*100:.1f}%")

    lens = [r["num_tokens_generated"] for r in results if "num_tokens_generated" in r]
    if lens:
        correct_lens = [
            r["num_tokens_generated"] for r in results
            if "num_tokens_generated" in r and r["correct"]
        ]
        incorrect_lens = [
            r["num_tokens_generated"] for r in results
            if "num_tokens_generated" in r and not r["correct"]
        ]
        print(f"Avg tokens generated: {sum(lens)/len(lens):.1f} (n={len(lens)})")
        if correct_lens:
            print(f"  correct:   {sum(correct_lens)/len(correct_lens):.1f} (n={len(correct_lens)})")
        if incorrect_lens:
            print(f"  incorrect: {sum(incorrect_lens)/len(incorrect_lens):.1f} (n={len(incorrect_lens)})")
    print(f"{'='*60}")

    # By level
    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        by_level[r["level"]]["correct"] += int(r["correct"])

    print(f"\n{'Level':<10} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 35)
    for level in sorted(by_level):
        d = by_level[level]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        print(f"Level {level:<4} {d['correct']:>8} {d['total']:>6} {acc:>7.1f}%")

    # By subject
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_subject[r["subject"]]["total"] += 1
        by_subject[r["subject"]]["correct"] += int(r["correct"])

    print(f"\n{'Subject':<25} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 50)
    for subj in sorted(by_subject):
        d = by_subject[subj]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        print(f"{subj:<25} {d['correct']:>8} {d['total']:>6} {acc:>7.1f}%")

    # Extraction failures
    no_answer = sum(1 for r in results if r["pred_answer"] is None)
    if no_answer:
        print(f"\nExtraction failures (no \\boxed{{}}): {no_answer}/{total}")


def save_results(eval_output: dict, output_dir: str):
    """Save full results and summary to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model_slug = eval_output["model"].replace("/", "_")

    # Full per-problem results
    results_path = os.path.join(output_dir, f"{model_slug}_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_output["results"], f, indent=2)

    # Summary
    results = eval_output["results"]
    total = len(results)
    correct = sum(r["correct"] for r in results)

    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        by_level[r["level"]]["correct"] += int(r["correct"])

    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_subject[r["subject"]]["total"] += 1
        by_subject[r["subject"]]["correct"] += int(r["correct"])

    lens = [r["num_tokens_generated"] for r in results if "num_tokens_generated" in r]
    correct_lens = [
        r["num_tokens_generated"] for r in results
        if "num_tokens_generated" in r and r["correct"]
    ]
    incorrect_lens = [
        r["num_tokens_generated"] for r in results
        if "num_tokens_generated" in r and not r["correct"]
    ]

    summary = {
        "model": eval_output["model"],
        "method": eval_output.get("method", "greedy"),
        "dataset_size": total,
        "overall_accuracy": correct / total if total else 0,
        "elapsed_s": eval_output["elapsed_s"],
        "max_tokens": eval_output.get("max_tokens"),
        "avg_tokens_generated": sum(lens) / len(lens) if lens else None,
        "avg_tokens_correct": (
            sum(correct_lens) / len(correct_lens) if correct_lens else None
        ),
        "avg_tokens_incorrect": (
            sum(incorrect_lens) / len(incorrect_lens) if incorrect_lens else None
        ),
        "by_level": {
            str(k): {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in sorted(by_level.items())
        },
        "by_subject": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in sorted(by_subject.items())
        },
        "extraction_failures": sum(
            1 for r in results if r["pred_answer"] is None
        ),
    }
    if "power_smc_config" in eval_output:
        summary["power_smc_config"] = eval_output["power_smc_config"]
    summary_path = os.path.join(output_dir, f"{model_slug}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on math benchmarks")
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local checkpoint path",
    )
    parser.add_argument(
        "--dataset", default="math500", choices=list(DATASET_REGISTRY_EVAL.keys()),
        help="Benchmark dataset to evaluate on",
    )
    parser.add_argument(
        "--levels", nargs="*", type=int, default=None,
        help="Filter to specific MATH difficulty levels (1-5)",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--prompt_mode", type=str, default="chat",
                        choices=["chat", "raw"],
                        help="Prompt format: 'chat' uses system+user chat template, "
                             "'raw' uses plain CoT string (matches Power-SMC reference for base models)")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on a random subset of N samples (useful for quick tests)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection")
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="Load chat template from this model (e.g. the instruct variant) "
                             "for base models that lack one")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="For models with a toggleable thinking mode (e.g. Qwen3): "
                            "pass --enable-thinking or --no-enable-thinking to override "
                            "the template default. Leave unset to use the model default.") 

    # Power-SMC options
    parser.add_argument("--power_smc", action="store_true",
                        help="Also evaluate using Power-SMC on the same samples")
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="Power exponent for power sampling")
    parser.add_argument("--smc_particles", type=int, default=64,
                        help="Number of Power-SMC particles")
    parser.add_argument("--smc_ess_threshold", type=float, default=0.5,
                        help="Resample when ESS drops below this fraction of particles")
    parser.add_argument("--smc_block_size", type=int, default=64,
                        help="Check ESS every N generated tokens")
    parser.add_argument("--smc_alpha_ramp_tokens", type=int, default=400,
                        help="Linearly ramp alpha over the first N generated tokens")
    parser.add_argument("--smc_min_new_tokens", type=int, default=100,
                        help="Suppress EOS for this many generated tokens")
    parser.add_argument("--smc_top_k", type=int, default=0,
                        help="Top-k truncation for the Power-SMC proposal; 0 disables it")
    parser.add_argument("--smc_top_p", type=float, default=0.9,
                        help="Nucleus truncation for the Power-SMC proposal; 1.0 disables it")
    parser.add_argument("--smc_repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty for the Power-SMC proposal")
    parser.add_argument("--no_smc_stop_on_boxed", action="store_true",
                        help="Do not stop Power-SMC particles after a non-empty boxed answer")
    parser.add_argument("--no_smc_cow_cache", action="store_true",
                        help="Disable copy-on-write cache compression after SMC resampling")
    parser.add_argument("--no_smc_shared_prompt_cache", action="store_true",
                        help="Process the prompt separately for every SMC particle")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"],
                        help="Model dtype (default: bfloat16)")
    

    args = parser.parse_args()

    # Load dataset
    loader = DATASET_REGISTRY_EVAL[args.dataset]
    problems = loader(levels=args.levels)
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    if args.levels:
        print(f"  Filtered to levels: {args.levels}")

    # Subset selection
    if args.num_samples is not None and args.num_samples < len(problems):
        import random
        random.seed(args.seed)
        problems = random.sample(problems, args.num_samples)
        print(f"  Subsampled to {len(problems)} problems (seed={args.seed})")

    # Load chat template tokenizer if specified
    chat_template_tokenizer = None
    if args.chat_template_model:
        from transformers import AutoTokenizer
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model, trust_remote_code=True
        )
        print(f"Using chat template from: {args.chat_template_model}")

    # Evaluate model
    model_slug = args.model.replace("/", "_")
    output_dir = args.output_dir+'/'+args.dataset+'/'+model_slug

    # Baseline (vLLM)
    eval_output = evaluate_model(
        model_name=args.model,
        problems=problems,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        chat_template_tokenizer=chat_template_tokenizer,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        prompt_mode=args.prompt_mode,
    )
    print_report(eval_output)
    save_results(eval_output, output_dir)

    # Power-SMC
    if args.power_smc:
        smc_output = evaluate_model_power_smc(
            model_name=args.model,
            problems=problems,
            max_tokens=args.max_tokens,
            alpha=args.alpha,
            n_particles=args.smc_particles,
            ess_threshold=args.smc_ess_threshold,
            block_size=args.smc_block_size,
            alpha_ramp_tokens=args.smc_alpha_ramp_tokens,
            min_new_tokens=args.smc_min_new_tokens,
            top_k=args.smc_top_k,
            top_p=args.smc_top_p,
            repetition_penalty=args.smc_repetition_penalty,
            chat_template_tokenizer=chat_template_tokenizer,
            enable_thinking=args.enable_thinking,
            dtype=args.dtype,
            stop_on_boxed=not args.no_smc_stop_on_boxed,
            use_cow_cache=not args.no_smc_cow_cache,
            prompt_mode=args.prompt_mode,
            shared_prompt_cache=not args.no_smc_shared_prompt_cache,
        )
        print_report(smc_output)
        smc_dir = output_dir + "/" + smc_output["method"]
        save_results(smc_output, smc_dir)


if __name__ == "__main__":
    main()
