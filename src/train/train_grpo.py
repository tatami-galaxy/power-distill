"""
GRPO training using TRL's GRPOTrainer.

Outcome-based reward: +1 if the model's boxed answer matches the gold answer, 0 otherwise.

    # Multi GPU
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run accelerate launch \
        --config_file configs/ddp_4.yaml \
        -m src.train.train_grpo --model Qwen/Qwen3-4B \
        --dataset deepmath --max_steps 1000 --logging_steps 10 \
        --save_steps 50 --gradient_accumulation_steps 4 \
        --save_total_limit 3 --per_device_batch_size 1 \
        --lr_scheduler_type constant 
"""

import argparse
import json
import os

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.utils import extract_boxed_answer, is_equiv, DATASET_REGISTRY_TRAIN


# ---------------------------------------------------------------------------
# Reward function (with all-wrong tracking)
# ---------------------------------------------------------------------------

class AccuracyRewardWithBuffer:
    """Outcome-based reward that also tracks prompts where all rollouts fail.

    When all `num_generations` rollouts for a prompt score 0, the prompt and
    gold answer are appended to a JSONL buffer file for later analysis /
    power-sampling recovery.
    """

    __name__ = "accuracy_reward"

    def __init__(self, num_generations: int, buffer_path: str):
        self.num_generations = num_generations
        self.buffer_path = buffer_path
        self.total_prompts = 0
        self.all_wrong_prompts = 0

        # Ensure the buffer directory exists and start fresh
        os.makedirs(os.path.dirname(buffer_path) or ".", exist_ok=True)
        if os.path.exists(buffer_path):
            os.remove(buffer_path)

    def __call__(self, completions, answer, prompts=None, **kwargs):
        rewards = []
        for completion, gold in zip(completions, answer):
            if isinstance(completion, list):
                text = completion[-1]["content"]
            else:
                text = completion
            pred = extract_boxed_answer(text)
            if pred is not None and is_equiv(pred, gold):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        # Group by prompt (every num_generations entries) and detect all-wrong
        G = self.num_generations
        for i in range(0, len(rewards), G):
            group_rewards = rewards[i : i + G]
            self.total_prompts += 1
            if sum(group_rewards) == 0.0:
                self.all_wrong_prompts += 1
                # Save prompt + answer for power-sampling recovery
                entry = {
                    "prompt": prompts[i] if prompts is not None else None,
                    "answer": answer[i],
                    "step": kwargs.get("trainer_state", None)
                    and kwargs["trainer_state"].global_step,
                }
                with open(self.buffer_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            # Log running stats every 100 prompts
            if self.total_prompts % 100 == 0:
                pct = 100 * self.all_wrong_prompts / self.total_prompts
                print(
                    f"[AllWrong] {self.all_wrong_prompts}/{self.total_prompts} "
                    f"prompts ({pct:.1f}%) had zero correct rollouts"
                )

        return rewards


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def format_grpo(example):
    """Format a NuminaMath/Polaris example for GRPO.

    Returns a dict with:
    - prompt: chat messages (system + user) for the model to complete
    - answer: gold answer string (passed to reward function as kwarg)
    """
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
    ]
    return {"prompt": prompt, "answer": example["answer"]}



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if args.chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(args.chat_template_model)
        tokenizer.chat_template = template_tok.chat_template
        print(f"Using chat template from: {args.chat_template_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA config
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    # Load and format dataset
    loader = DATASET_REGISTRY_TRAIN[args.dataset]
    loader_kwargs = dict(max_samples=args.max_samples, seed=args.seed)
    if args.dataset == "deepmath":
        loader_kwargs["explode_solutions"] = False
    if args.dataset == "numinamath" and args.sources:
        loader_kwargs["sources"] = args.sources
    if args.dataset == "polaris" and args.difficulty:
        loader_kwargs["difficulty"] = args.difficulty
    ds = loader(**loader_kwargs)
    print(f"Loaded {len(ds)} training examples from {args.dataset}")

    ds = ds.map(
        format_grpo,
        remove_columns=[c for c in ds.column_names if c not in ["answer"]],
        num_proc=4,
    )

    # Training config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        #max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        beta=args.beta,
        # vLLM
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        # Logging / saving
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to="tensorboard",
    )

    # Reward function with all-wrong buffer
    buffer_path = os.path.join(args.output_dir, "all_wrong_buffer.jsonl")
    reward_fn = AccuracyRewardWithBuffer(
        num_generations=args.num_generations,
        buffer_path=buffer_path,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Print training plan
    num_devices = max(torch.cuda.device_count(), 1)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = len(ds) // effective_batch
    print(f"\n{'='*60}")
    print(f"Training plan (GRPO):")
    print(f"  Dataset size:        {len(ds)}")
    print(f"  Devices:             {num_devices}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
    print(f"  Effective batch:     {effective_batch}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Max steps:           {args.max_steps}")
    print(f"  Num generations:     {args.num_generations}")
    print(f"  Max completion len:  {args.max_completion_length}")
    print(f"  Temperature:         {args.temperature}")
    print(f"  Beta (KL coeff):     {args.beta}")
    print(f"  Log every:           {args.logging_steps} steps")
    print(f"  Save every:          {args.save_steps} steps")
    print(f"  Warmup steps:        {args.warmup_steps}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  vLLM:                {args.use_vllm}")
    if args.use_lora:
        print(f"  LoRA r:              {args.lora_r}")
        print(f"  LoRA alpha:          {args.lora_alpha}")
        print(f"  LoRA dropout:        {args.lora_dropout}")
        print(f"  LoRA targets:        {args.lora_target_modules}")
    print(f"{'='*60}\n")

    # Train
    trainer.train()

    # Print all-wrong summary
    if reward_fn.total_prompts > 0:
        pct = 100 * reward_fn.all_wrong_prompts / reward_fn.total_prompts
        print(f"\n{'='*60}")
        print(f"All-wrong summary:")
        print(f"  Total prompts seen:   {reward_fn.total_prompts}")
        print(f"  All-wrong prompts:    {reward_fn.all_wrong_prompts} ({pct:.1f}%)")
        print(f"  Buffer saved to:      {buffer_path}")
        print(f"{'='*60}\n")

    # Save training config for reproducibility
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO on NuminaMath-1.5 or Polaris-Dataset-53K")

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/rl")

    # Data
    parser.add_argument("--dataset", type=str, default="numinamath",
                        choices=list(DATASET_REGISTRY_TRAIN.keys()),
                        help="Dataset to train on (default: numinamath)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="Filter to specific NuminaMath sources (e.g. olympiads cn_k12)",
    )
    parser.add_argument(
        "--difficulty", nargs="*", default=None,
        help="Filter Polaris by difficulty (e.g. 1/8 2/8 3/8)",
    )

    # GRPO
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of completions per prompt")
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty coefficient (0 = no KL)")

    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", nargs="*",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="LoRA target modules")

    # vLLM
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for faster generation (colocate mode)")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.3,
                        help="GPU memory fraction for vLLM (colocate mode)")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="HF model to borrow chat template from (e.g. instruct variant for a base model)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
