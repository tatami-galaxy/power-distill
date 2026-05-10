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

from datasets import Dataset

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

    def _completion_to_text(self, completion):
        if isinstance(completion, list):
            return completion[-1]["content"]
        return completion

    @staticmethod
    def _prompt_key(prompt):
        """Hashable key from a chat-message prompt list."""
        if prompt is None:
            return None
        for msg in prompt:
            if msg["role"] == "user":
                return msg["content"]
        return str(prompt)

    def __call__(self, completions, answer, prompts=None, **kwargs):
        rewards = []
        for completion, gold in zip(completions, answer):
            text = self._completion_to_text(completion)
            pred = extract_boxed_answer(text)
            if pred is not None and is_equiv(pred, gold):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        # Group by prompt identity (robust to DDP splitting / ordering)
        from collections import OrderedDict
        groups = OrderedDict()  # key -> {prompt, answer, rewards, completions}
        for idx in range(len(rewards)):
            prompt = prompts[idx] if prompts is not None else None
            key = (self._prompt_key(prompt), answer[idx])
            if key not in groups:
                groups[key] = {
                    "prompt": prompt,
                    "answer": answer[idx],
                    "rewards": [],
                    "completions": [],
                }
            groups[key]["rewards"].append(rewards[idx])
            groups[key]["completions"].append(
                self._completion_to_text(completions[idx])
            )

        step = (
            kwargs.get("trainer_state", None)
            and kwargs["trainer_state"].global_step
        )
        for group in groups.values():
            self.total_prompts += 1
            if sum(group["rewards"]) == 0.0:
                self.all_wrong_prompts += 1
                entry = {
                    "prompt": group["prompt"],
                    "answer": group["answer"],
                    "completions": group["completions"],
                    "step": step,
                }
                with open(self.buffer_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            if self.total_prompts % 100 == 0:
                pct = 100 * self.all_wrong_prompts / self.total_prompts
                print(
                    f"[AllWrong] {self.all_wrong_prompts}/{self.total_prompts} "
                    f"prompts ({pct:.1f}%) had zero correct rollouts"
                )

        return rewards


# ---------------------------------------------------------------------------
# Recovery: GRPOTrainer that trains on pre-computed rollout batches
# ---------------------------------------------------------------------------

def _pad_tensors(tensors, padding_value, side="right"):
    """Pad a list of 1-D tensors to equal length."""
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), padding_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        if side == "right":
            out[i, : t.size(0)] = t
        else:
            out[i, max_len - t.size(0) :] = t
    return out


def load_recovered_dataset(path: str, num_generations: int) -> Dataset:
    """Load recovered_buffer.jsonl into a HF Dataset for recovery training.

    Each entry has: prompt (chat messages), answer, completions (variable-length
    wrong rollouts), power_response (1 correct).  We pad/truncate wrong
    completions to exactly G-1 by repeating if needed, then append the correct
    one so every group has exactly G completions.
    """
    G = num_generations
    rows = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            wrongs = entry["completions"]
            # Pad to G-1 by cycling, or truncate
            if len(wrongs) == 0:
                continue  # can't build a group with no wrong completions
            if len(wrongs) < G - 1:
                repeats = (G - 1) // len(wrongs) + 1
                wrongs = (wrongs * repeats)[: G - 1]
            else:
                wrongs = wrongs[: G - 1]
            stored = wrongs + [entry["power_response"]]
            rows.append({
                "prompt": entry["prompt"],
                "answer": entry["answer"],
                "stored_completions": stored,
            })
    print(f"Loaded {len(rows)} recovered entries from {path}")
    return Dataset.from_list(rows)


class GRPOWithRecovery(GRPOTrainer):
    """GRPOTrainer that can train on pre-computed recovery rollouts.

    When the dataset contains a ``stored_completions`` column, generation is
    skipped and the stored completions are used directly.  Rewards are set to
    [0, ..., 0, 1] (last completion is the power-sampled correct one) and
    advantages are computed as usual.

    NOTE: currently single-GPU only for recovery mode.
    """

    def _generate_and_score_completions(self, inputs):
        # Normal GRPO path when no stored completions
        if "stored_completions" not in inputs[0]:
            return super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        G = self.num_generations
        B = len(inputs)

        # --- 1. Tokenize prompts (apply chat template, repeat G times) ---
        prompt_ids_list = []
        for inp in inputs:
            text = self._tokenizer.apply_chat_template(
                inp["prompt"], tokenize=False, add_generation_prompt=True
            )
            ids = self._tokenizer.encode(text)
            for _ in range(G):
                prompt_ids_list.append(torch.tensor(ids, dtype=torch.long))

        # --- 2. Tokenize stored completions (G per prompt) ---
        completion_ids_list = []
        for inp in inputs:
            for comp_text in inp["stored_completions"]:
                ids = self._tokenizer.encode(
                    comp_text, add_special_tokens=False
                )
                ids = ids[: self.max_completion_length]
                completion_ids_list.append(torch.tensor(ids, dtype=torch.long))

        # --- 3. Pad into tensors ---
        pad_id = self._tokenizer.pad_token_id
        prompt_ids = _pad_tensors(prompt_ids_list, pad_id, side="left").to(device)
        prompt_mask = (prompt_ids != pad_id).long()
        completion_ids = _pad_tensors(completion_ids_list, pad_id, side="right").to(device)
        completion_mask = (completion_ids != pad_id).long()

        # --- 4. Forward pass → old_per_token_logps ---
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, input_ids, attention_mask, logits_to_keep,
                batch_size=self.args.per_device_train_batch_size,
            )

        # --- 5. Rewards [0,...,0,1] and group-relative advantages ---
        rewards = torch.zeros(B * G, device=device)
        for i in range(B):
            rewards[i * G + G - 1] = 1.0

        mean_r = rewards.view(B, G).mean(dim=1).repeat_interleave(G)
        std_r = rewards.view(B, G).std(dim=1).repeat_interleave(G)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # --- 6. Ref model logps (if KL penalty) ---
        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, input_ids, attention_mask,
                        logits_to_keep,
                        batch_size=self.args.per_device_train_batch_size,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    adapter = "ref" if hasattr(model, "peft_config") and "ref" in model.peft_config else None
                    from contextlib import nullcontext
                    ctx = nullcontext()
                    if adapter is not None:
                        from trl.trainer.utils import use_adapter
                        ctx = use_adapter(model, adapter_name=adapter)
                    with ctx:
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model, input_ids, attention_mask,
                            logits_to_keep,
                            batch_size=self.args.per_device_train_batch_size,
                        )

        # --- 7. Metrics ---
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode][f"rewards/{self.reward_func_names[0]}/mean"].append(
            rewards.mean().item()
        )
        self._metrics[mode][f"rewards/{self.reward_func_names[0]}/std"].append(
            rewards.std().item()
        )
        self._metrics[mode]["frac_reward_zero_std"].append(0.0)

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": B * G,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def format_grpo(example):
    """Format a example for GRPO.

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

    # Load dataset (normal or recovery mode)
    recovery_mode = args.recovered_buffer is not None
    if recovery_mode:
        ds = load_recovered_dataset(args.recovered_buffer, args.num_generations)
        print(f"[Recovery] Training on {len(ds)} recovered rollout batches")
    else:
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

    # Trainer (use recovery subclass when training on recovered rollouts)
    trainer_cls = GRPOWithRecovery if recovery_mode else GRPOTrainer
    trainer = trainer_cls(
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
    print(f"Training plan ({'Recovery GRPO' if recovery_mode else 'GRPO'}):")
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

    # Recovery (Phase 3)
    parser.add_argument("--recovered_buffer", type=str, default=None,
                        help="Path to recovered_buffer.jsonl from power sampling. "
                             "Trains on recovered rollouts instead of generating new ones.")

    args = parser.parse_args()

    # add model name to output_dir
    args.output_dir = args.output_dir+"/"+args.dataset+"/"+args.model.replace("/","_")
    
    train(args)


if __name__ == "__main__":
    main()
