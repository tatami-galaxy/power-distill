"""
SDFT: Self-Distillation Fine-Tuning on math datasets (NuminaMath-1.5 / competition_math).

On-policy algorithm from "Self-Distillation Fine-Tuning" (arXiv:2601.19897).

The model simultaneously serves as student and teacher via in-context learning:
  - Student: receives (system + question), generates response on-policy
  - Teacher: receives (system + question + expert_solution), provides target distribution

Loss: analytic KL divergence over completion tokens (per-sequence normalized):
    forward (default, matches reference code):
                      D_KL(πt ‖ πθ)  = Σ_v πt(v) · (log πt(v) − log πθ(v))
    reverse (matches paper equations):
                      D_KL(πθ ‖ πt)  = Σ_v πθ(v) · (log πθ(v) − log πt(v))

Teacher weights updated via EMA after each optimizer step: φ ← α·θ + (1−α)·φ

~2.5× the compute of SFT (on-policy generation + two forward passes per step).

Usage:

    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run accelerate launch \\
        --config_file configs/multi_gpu_4.yaml -m \\
        scripts.train_sdft --model Qwen/Qwen3-4B \\
        --dataset deepmath --save_steps 50 \\
        --gradient_accumulation_steps 4 --save_total_limit 3 \\
        --max_prompt_length 8192 --per_device_batch_size 2 \\
        --lr_scheduler_type constant
"""

import argparse
import contextlib
import copy
import json
import os
import warnings

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import (
    load_deepmath,
    load_openthoughts,
    extract_boxed_answer, is_equiv
)


def _unwrap_model(model):
    """Unwrap DDP/FSDP to access the underlying model (e.g. for .generate())."""
    if isinstance(model, DDP):
        return model.module
    if hasattr(model, "module"):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Prompt templates (from SDFT paper, Appendix A)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)

TEACHER_TEMPLATE = (
    "{question}\n\n"
    "This is an example for a response to the question:\n"
    "{demonstration}\n\n"
    "Now answer with a response of your own, including the thinking process:"
)


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def format_sdft(example):
    """Create student and teacher message lists for one NuminaMath example.

    Student sees only the question.
    Teacher sees the question + expert solution via the ICL template.
    """
    question = example["problem"]
    solution = example["solution"].strip()
    if "\\boxed" not in solution:
        solution += f"\n\nThe answer is $\\boxed{{{example['answer']}}}$."

    student_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    teacher_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEACHER_TEMPLATE.format(
            question=question, demonstration=solution
        )},
    ]
    return {"student_messages": student_messages, "teacher_messages": teacher_messages}


class SDFTDataCollator:
    """Apply chat template, tokenize, and pad student and teacher prompts."""

    def __init__(self, tokenizer, max_prompt_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __call__(self, features):
        def _encode(key):
            texts = [
                self.tokenizer.apply_chat_template(
                    f[key], tokenize=False, add_generation_prompt=True
                )
                for f in features
            ]
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
            )
            return texts, enc

        s_texts, s = _encode("student_messages")
        _, t = _encode("teacher_messages")
        return {
            "student_input_ids": s["input_ids"],
            "student_attention_mask": s["attention_mask"],
            "teacher_input_ids": t["input_ids"],
            "teacher_attention_mask": t["attention_mask"],
            "student_prompt_texts": s_texts,  # raw strings for vLLM
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_padded_batch(
    prompt_ids: list[torch.Tensor],
    completion_ids: list[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Left-pad [prompt ‖ completion] so all completions right-align.

    Layout per row i: ``[PAD × (max_len − Lp_i − Lc_i), prompt_i, completion_i]``.
    With completions flush to the right edge, ``logits_to_keep = max_Lc + 1``
    passed to the forward skips projecting non-completion hidden states
    through the (huge) lm_head.

    comp_mask aligns with ``model(...).logits[:, -(max_Lc+1):-1, :]``:
    comp_mask[i, t] = True iff that logit predicts a completion token for item i.

    Returns:
        input_ids : [B, max_len]
        attn_mask : [B, max_len]
        comp_mask : [B, max_Lc]  bool, right-aligned
        max_Lc    : int
    """
    B = len(prompt_ids)
    Lps = [p.shape[0] for p in prompt_ids]
    Lcs = [c.shape[0] for c in completion_ids]
    lens = [lp + lc for lp, lc in zip(Lps, Lcs)]
    max_len = max(lens)
    max_Lc = max(Lcs) if Lcs else 0

    ids  = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    comp = torch.zeros(B, max_Lc, dtype=torch.bool)

    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        L = Lps[i] + Lcs[i]
        ids[i, max_len - L:] = torch.cat([p, c])
        mask[i, max_len - L:] = 1
        if Lcs[i] > 0:
            comp[i, max_Lc - Lcs[i]:] = True

    return ids, mask, comp, max_Lc


def _apply_skip_mask(
    per_token_kl: torch.Tensor,
    comp_mask: torch.Tensor,
    skip_n: int,
) -> torch.Tensor:
    """Zero out the first skip_n completion tokens per sequence.

    Reference impl uses skip_n=3 to suppress noisy KL on structural tokens
    (e.g. <think>, opening newlines) that have high variance.
    """
    # comp_mask: [B, max_Lc] bool, right-aligned.
    counts = comp_mask.sum(dim=1)                               # [B] tokens per sequence
    offsets = counts.cumsum(dim=0) - counts                     # [B] start offset in flat [N]
    # Repeat each offset by its count to get the sequence-start for every flat token.
    seq_offsets = offsets.repeat_interleave(counts)              # [N]
    flat_idx = torch.arange(per_token_kl.shape[0], device=per_token_kl.device)
    within_seq_idx = flat_idx - seq_offsets                     # position within its sequence
    weight = (within_seq_idx >= skip_n).to(per_token_kl.dtype)
    return per_token_kl * weight


# ---------------------------------------------------------------------------
# EMA callback — teacher = exponential moving average of student
# ---------------------------------------------------------------------------

class EMACallback(TrainerCallback):
    """After each optimizer step: φ ← α·θ + (1−α)·φ."""

    def __init__(self, teacher_model: AutoModelForCausalLM, ema_alpha: float = 0.01):
        self.teacher_model = teacher_model
        self.ema_alpha = ema_alpha

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Move teacher to the same device as the student (set by Trainer).
        device = next(model.parameters()).device
        self.teacher_model.to(device)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        alpha = self.ema_alpha
        # Under FSDP, summon full (unsharded) params so each rank sees the
        # original 2-D shapes and applies the identical EMA update to its
        # replicated teacher copy.
        summon_ctx = (
            FSDP.summon_full_params(model, writeback=False, recurse=True)
            if isinstance(model, FSDP)
            else contextlib.nullcontext()
        )
        with summon_ctx:
            student = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), self.teacher_model.parameters()):
                    pt.data.mul_(1.0 - alpha).add_(ps.data, alpha=alpha)


# ---------------------------------------------------------------------------
# vLLM weight-sync callback — save checkpoint so vLLM server can reload
# ---------------------------------------------------------------------------

class VLLMWeightSyncCallback(TrainerCallback):
    """Every *sync_steps* optimizer steps, save the student weights to
    *checkpoint_dir* and (optionally) tell a running vLLM server to reload.

    If a vLLM ``OpenAI`` client is provided the callback will POST to the
    ``/v1/load_checkpoint`` endpoint (supported by vLLM ≥ 0.6).  Otherwise it
    just saves the checkpoint and prints a reminder to restart the server.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        sync_steps: int = 512,
        vllm_client=None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.sync_steps = sync_steps
        self.vllm_client = vllm_client
        if int(os.environ.get("RANK", 0)) == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.sync_steps != 0:
            return

        rank = int(os.environ.get("RANK", 0))
        # Under FSDP we must all-gather full params across ranks, then save
        # from rank 0 only. Under DDP, params are already replicated.
        summon_ctx = (
            FSDP.summon_full_params(model, writeback=False, recurse=True, offload_to_cpu=True, rank0_only=True)
            if isinstance(model, FSDP)
            else contextlib.nullcontext()
        )
        with summon_ctx:
            if rank == 0:
                student = model.module if hasattr(model, "module") else model
                student.save_pretrained(self.checkpoint_dir)
                print(f"[VLLMSync step {state.global_step}] Saved weights → {self.checkpoint_dir}")

        if rank != 0:
            return

        if self.vllm_client is not None:
            try:
                # vLLM ≥ 0.6 exposes /v1/load_checkpoint on the OpenAI server.
                import httpx
                base = self.vllm_client.base_url
                url = f"{base}load_checkpoint"
                resp = httpx.post(url, json={"checkpoint_dir": self.checkpoint_dir}, timeout=120)
                resp.raise_for_status()
                print(f"[VLLMSync step {state.global_step}] Server reloaded weights")
            except Exception as e:
                warnings.warn(
                    f"[VLLMSync step {state.global_step}] Could not reload "
                    f"vLLM server ({e}). Restart manually or rely on IS correction."
                )


# ---------------------------------------------------------------------------
# SDFT Trainer
# ---------------------------------------------------------------------------

class SDFTTrainer(Trainer):
    """On-policy self-distillation trainer.

    Each training step:
      1. Generate completion  y ~ πθ(·|CtxS(x))  [no grad]
      2. Teacher forward on  (CtxT(x, c) ‖ y)    [no grad]
      3. Student forward on  (CtxS(x) ‖ y)        [with grad]
      4. Loss = analytic reverse KL  D_KL(πθ ‖ π_teacher)  per completion token
    """

    def __init__(
        self,
        *args,
        teacher_model: AutoModelForCausalLM,
        sdft_tokenizer: AutoTokenizer,
        max_new_tokens: int = 512,
        skip_first_n_tokens: int = 3,
        kl_direction: str = "forward",
        kl_topk: int | None = None,
        # vLLM options
        use_vllm: bool = False,
        vllm_port: int = 8000,
        importance_sampling_correction: bool = True,
        importance_sampling_cap: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.sdft_tokenizer = sdft_tokenizer
        self.max_new_tokens = max_new_tokens
        self.skip_first_n_tokens = skip_first_n_tokens
        self.kl_direction = kl_direction
        self.kl_topk = kl_topk
        self.use_vllm = use_vllm
        self.importance_sampling_correction = importance_sampling_correction
        self.importance_sampling_cap = importance_sampling_cap

        if use_vllm:
            if not _HAS_OPENAI:
                raise ImportError(
                    "vLLM server mode requires the `openai` package. "
                    "Install it with: pip install openai"
                )
            self.vllm_client = _OpenAI(
                base_url=f"http://localhost:{vllm_port}/v1",
                api_key="unused",  # vLLM doesn't require a real key
            )

    # -- vLLM generation ----------------------------------------------------

    def _generate_vllm(
        self,
        prompt_texts: list[str],
    ) -> tuple[list[torch.Tensor], list[list[float]]]:
        """Generate completions via vLLM OpenAI-compatible server.

        Returns:
            completions  : list of 1-D token-id tensors (one per prompt)
            vllm_logprobs: list of per-token log-prob lists (one per prompt,
                           each inner list has length == len(completion))
        """
        # Use the model name the vLLM server was started with.  The /v1/models
        # endpoint always lists exactly one entry for single-model serving.
        models = self.vllm_client.models.list()
        model_name = models.data[0].id

        response = self.vllm_client.completions.create(
            model=model_name,
            prompt=prompt_texts,
            max_tokens=self.max_new_tokens,
            temperature=1.0,  # on-policy sampling
            logprobs=1,       # return per-token log-probs
        )

        tokenizer = self.sdft_tokenizer
        completions: list[torch.Tensor] = []
        vllm_logprobs: list[list[float]] = []

        # Responses arrive in the same order as prompts (index field).
        sorted_choices = sorted(response.choices, key=lambda c: c.index)
        for choice in sorted_choices:
            text = choice.text
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            completions.append(torch.tensor(token_ids, dtype=torch.long))

            # Extract log-probs returned by vLLM.
            lps: list[float] = []
            if choice.logprobs and choice.logprobs.tokens:
                lps = [
                    lp for lp in choice.logprobs.token_logprobs
                    if lp is not None  # first token may be None
                ]
            # Pad / truncate to match token count (re-tokenisation may
            # differ by ±1 from vLLM's tokeniser).
            while len(lps) < len(token_ids):
                lps.append(0.0)
            lps = lps[: len(token_ids)]
            vllm_logprobs.append(lps)

        return completions, vllm_logprobs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        s_ids  = inputs["student_input_ids"]       # [B, Ls]
        s_mask = inputs["student_attention_mask"]   # [B, Ls]
        t_ids  = inputs["teacher_input_ids"]        # [B, Lt]
        t_mask = inputs["teacher_attention_mask"]   # [B, Lt]

        B      = s_ids.shape[0]
        device = s_ids.device
        pad_id = self.sdft_tokenizer.pad_token_id

        # ── 1. On-policy generation ──────────────────────────────────────────
        vllm_token_logprobs = None  # populated only when use_vllm is True

        if self.use_vllm:
            # ── 1a. Generate via vLLM server ─────────────────────────────────
            prompt_texts = inputs["student_prompt_texts"]
            completions, vllm_token_logprobs = self._generate_vllm(prompt_texts)
            student_prompts = [s_ids[i][s_mask[i].bool()].cpu() for i in range(B)]
            teacher_prompts = [t_ids[i][t_mask[i].bool()].cpu() for i in range(B)]
        else:
            # ── 1b. Generate via HF model (batched, left-padded) ─────────────
            model.eval()

            seq_lens = s_mask.sum(dim=1)
            Ls = s_ids.shape[1]
            left_ids  = s_ids.new_full((B, Ls), pad_id)
            left_mask = torch.zeros_like(s_mask)
            for i in range(B):
                L = seq_lens[i]
                left_ids[i, Ls - L:]  = s_ids[i, :L]
                left_mask[i, Ls - L:] = 1

            gen_ctx = (
                FSDP.summon_full_params(model, writeback=False, recurse=True)
                if isinstance(model, FSDP)
                else contextlib.nullcontext()
            )
            with torch.no_grad(), gen_ctx:
                gen_out = _unwrap_model(model).generate(
                    left_ids,
                    attention_mask=left_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    pad_token_id=pad_id,
                    eos_token_id=self.sdft_tokenizer.eos_token_id,
                )  # [B, Ls + gen_len]

            model.train()

            completions, student_prompts, teacher_prompts = [], [], []
            prompt_len = Ls
            for i in range(B):
                comp = gen_out[i, prompt_len:]
                eos_pos = (comp == self.sdft_tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    comp = comp[: eos_pos[0] + 1]
                non_pad = (comp != pad_id).nonzero(as_tuple=True)[0]
                if non_pad.numel() > 0:
                    comp = comp[: non_pad[-1] + 1]
                else:
                    comp = comp[:0]
                completions.append(comp.cpu())
                student_prompts.append(s_ids[i][s_mask[i].bool()].cpu())
                teacher_prompts.append(t_ids[i][t_mask[i].bool()].cpu())

        # ── 2. Build left-padded full sequences (prompt ‖ completion) ────────
        sf_ids, sf_mask, comp_s, max_Lc = _build_padded_batch(student_prompts, completions, pad_id)
        tf_ids, tf_mask, comp_t, _      = _build_padded_batch(teacher_prompts, completions, pad_id)

        sf_ids  = sf_ids.to(device);  sf_mask = sf_mask.to(device)
        tf_ids  = tf_ids.to(device);  tf_mask = tf_mask.to(device)
        comp_s  = comp_s.to(device);  comp_t  = comp_t.to(device)
        logits_to_keep = max_Lc + 1

        # ── 3. Teacher forward (no grad) ──────────────────────────────────────
        with torch.no_grad():
            t_logits = self.teacher_model(
                input_ids=tf_ids, attention_mask=tf_mask, use_cache=False,
                logits_to_keep=logits_to_keep,
            ).logits[:, :-1, :]  # [B, max_Lc, V]

        # ── 4. Student forward (with grad) ────────────────────────────────────
        s_logits = model(
            input_ids=sf_ids, attention_mask=sf_mask, use_cache=False,
            logits_to_keep=logits_to_keep,
        ).logits[:, :-1, :]  # [B, max_Lc, V]

        # ── 5. KL divergence per completion token ─────────────────────────────
        if self.kl_topk is not None:
            # Top-k KL: take student's top-k logits, gather teacher at same
            # indices, renormalize on the subset. Matches TRL's default.
            k = self.kl_topk
            s_topk_vals, topk_idx = s_logits.topk(k, dim=-1)         # [B, max_Lc, k]
            t_topk_vals = t_logits.gather(-1, topk_idx)              # [B, max_Lc, k]
            s_topk_log_p = s_topk_vals.float() - s_topk_vals.float().logsumexp(-1, keepdim=True)
            t_topk_log_p = t_topk_vals.float() - t_topk_vals.float().logsumexp(-1, keepdim=True)
            student_log_p = s_topk_log_p[comp_s]                     # [N, k]
            teacher_log_p = t_topk_log_p[comp_t]                     # [N, k]
        else:
            student_log_p = torch.log_softmax(s_logits[comp_s].float(), dim=-1)  # [N, V]
            teacher_log_p = torch.log_softmax(t_logits[comp_t].float(), dim=-1)  # [N, V]

        if self.kl_direction == "reverse":
            per_token_kl = (student_log_p.exp() * (student_log_p - teacher_log_p)).sum(-1)  # [N]
        else:
            per_token_kl = (teacher_log_p.exp() * (teacher_log_p - student_log_p)).sum(-1)  # [N]

        if self.skip_first_n_tokens > 0:
            per_token_kl = _apply_skip_mask(per_token_kl, comp_s, self.skip_first_n_tokens)

        # ── 6. Importance-sampling correction (vLLM only) ─────────────────────
        #  Tokens were sampled from π_vllm (possibly stale weights).
        #  IS weight = π_θ(y_t) / π_vllm(y_t),  applied per-sequence.
        if self.use_vllm and self.importance_sampling_correction and vllm_token_logprobs is not None:
            # Compute per-token student log-prob at the sampled token ids
            # directly from raw logits (independent of top-k path).
            comp_token_ids = torch.cat(completions).to(device)       # [N]
            s_logits_flat = s_logits[comp_s]                         # [N, V]
            s_sel = s_logits_flat.gather(-1, comp_token_ids.unsqueeze(-1)).squeeze(-1).float()
            s_lse = s_logits_flat.float().logsumexp(-1)
            student_selected_lp = s_sel - s_lse                      # [N]
            del s_logits_flat

            # Flatten vLLM log-probs to match.
            vllm_lp = torch.tensor(
                [lp for seq_lps in vllm_token_logprobs for lp in seq_lps],
                dtype=student_selected_lp.dtype, device=device,
            )  # [N]

            per_token_ratio = (student_selected_lp.detach() - vllm_lp).exp()

            # Per-sequence mean ratio, then expand back to per-token weights.
            offset = 0
            is_weights = []  # one scalar per sequence
            for row in comp_s:
                n = int(row.sum().item())
                if n > 0:
                    seq_ratio = per_token_ratio[offset:offset + n].mean()
                    is_weights.append(seq_ratio.clamp(max=self.importance_sampling_cap))
                offset += n

            # Apply per-sequence IS weight to per_token_kl.
            offset = 0
            for idx, row in enumerate(comp_s):
                n = int(row.sum().item())
                if n > 0:
                    per_token_kl[offset:offset + n] = per_token_kl[offset:offset + n] * is_weights[idx]
                offset += n

        # ── 7. Per-sequence average, then batch average ───────────────────────
        offset = 0
        seq_losses = []
        for row in comp_s:
            n = int(row.sum().item())
            if n > 0:
                seq_losses.append(per_token_kl[offset:offset + n].mean())
            offset += n
        loss = torch.stack(seq_losses).mean()

        return (loss, None) if return_outputs else loss


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args):

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(args.chat_template_model)
        tokenizer.chat_template = template_tok.chat_template
        print(f"Using chat template from: {args.chat_template_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
    )

    # ── Teacher (EMA copy, frozen, eval mode) ─────────────────────────────────
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # Output dir
    if args.demo_file:
        demo_name = os.path.splitext(os.path.basename(args.demo_file))[0]
        args.output_dir = f"{args.output_dir}/{demo_name}/{args.model.split('/')[-1]}"
    else:
        args.output_dir = f"{args.output_dir}/{args.dataset}/{args.model.split('/')[-1]}"

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.demo_file:
        from datasets import load_dataset as _load_ds
        ds = _load_ds("json", data_files=args.demo_file, split="train")
        if args.max_samples and args.max_samples < len(ds):
            ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))
        print(f"Loaded {len(ds)} demos from {args.demo_file}")
    elif args.dataset == "openthoughts":
        ds = load_openthoughts(max_samples=args.max_samples, seed=args.seed)
    elif args.dataset == "deepmath":
        ds = load_deepmath(max_samples=args.max_samples, seed=args.seed)
    else:
        raise ValueError(f"Provide --demo_file or a valid --dataset")
    if not args.demo_file:
        print(f"Loaded {len(ds)} training examples from {args.dataset}")

    train_ds = ds.map(
        format_sdft,
        remove_columns=ds.column_names,
        num_proc=4,
    )

    # Filter: drop examples where either prompt exceeds max_prompt_length.
    def _fits(example):
        for key in ("student_messages", "teacher_messages"):
            text = tokenizer.apply_chat_template(
                example[key], tokenize=False, add_generation_prompt=True
            )
            if len(tokenizer.encode(text)) > args.max_prompt_length:
                return False
        return True

    pre = len(train_ds)
    train_ds = train_ds.filter(_fits, num_proc=4)
    print(f"Filtered by max_prompt_length={args.max_prompt_length}: {pre} → {len(train_ds)}")


    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        optim=args.optim,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",  
        seed=args.seed,
        report_to="tensorboard",
        remove_unused_columns=False,  # must keep student_/teacher_ columns
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ema_cb = EMACallback(teacher_model, ema_alpha=args.ema_alpha)
    callbacks = [ema_cb]

    if args.use_vllm:
        vllm_ckpt_dir = os.path.join(args.output_dir, "vllm_sync")
        # The trainer creates the vllm_client in __init__; grab it after
        # trainer construction.  For now, pass None and patch below.
        vllm_sync_cb = VLLMWeightSyncCallback(
            checkpoint_dir=vllm_ckpt_dir,
            sync_steps=args.vllm_sync_steps,
            vllm_client=None,  # patched after trainer init
        )
        callbacks.append(vllm_sync_cb)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SDFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=SDFTDataCollator(tokenizer, max_prompt_length=args.max_prompt_length),
        callbacks=callbacks,
        # SDFT-specific
        teacher_model=teacher_model,
        sdft_tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        skip_first_n_tokens=args.skip_first_n_tokens,
        kl_direction=args.kl_direction,
        kl_topk=args.kl_topk,
        # vLLM
        use_vllm=args.use_vllm,
        vllm_port=args.vllm_port,
        importance_sampling_correction=args.importance_sampling_correction,
        importance_sampling_cap=args.importance_sampling_cap,
    )
    if args.use_vllm:
        vllm_sync_cb.vllm_client = trainer.vllm_client

    # ── Summary ───────────────────────────────────────────────────────────────
    num_devices = max(torch.cuda.device_count(), 1)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = len(train_ds) // effective_batch
    print(f"\n{'='*60}")
    print(f"SDFT training plan:")
    print(f"  Train size:          {len(train_ds)}")
    print(f"  Devices:             {num_devices}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
    print(f"  Effective batch:     {effective_batch}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Max steps:           {args.max_steps}")
    print(f"  Max new tokens:      {args.max_new_tokens}")
    print(f"  EMA alpha:           {args.ema_alpha}")
    print(f"  Skip first tokens:   {args.skip_first_n_tokens}")
    print(f"  KL direction:        {args.kl_direction}")
    print(f"  KL top-k:            {args.kl_topk}")
    print(f"  Learning rate:       {args.learning_rate}")
    if args.use_vllm:
        print(f"  vLLM port:           {args.vllm_port}")
        print(f"  vLLM sync steps:     {args.vllm_sync_steps}")
        print(f"  IS correction:       {args.importance_sampling_correction}")
        print(f"  IS cap:              {args.importance_sampling_cap}")
    print(f"{'='*60}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "final"))

    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDFT")

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/sdft")

    # Data
    parser.add_argument("--dataset", type=str, default="numinamath",
                        choices=["numinamath", "competition_math", "deepmath"],
                        help="Training dataset: 'numinamath' (AI-MO/NuminaMath-1.5), "
                             "'competition_math' (qwedsacf/competition_math), "
                             "or 'deepmath' (zwhe99/DeepMath-103K)")
    parser.add_argument("--demo_file", type=str, default=None,
                        help="JSONL file with power-sampled demos (problem, solution, answer). "
                             "Overrides --dataset. Auto-filters to correct=True.")
    parser.add_argument("--max_samples", type=int, default=None)

    # Generation (on-policy sampling)
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max completion length for on-policy generation")
    parser.add_argument("--max_prompt_length", type=int, default=4096,
                        help="Max tokenized length for student/teacher prompts")

    # SDFT-specific hyperparameters
    parser.add_argument("--ema_alpha", type=float, default=0.01,
                        help="EMA rate for teacher update: φ ← α·θ + (1−α)·φ")
    parser.add_argument("--skip_first_n_tokens", type=int, default=3,
                        help="Skip first n completion tokens in KL loss (structural tokens)")
    parser.add_argument("--kl_direction", type=str, default="forward",
                        choices=["reverse", "forward"],
                        help="KL direction: 'forward' matches reference codebase default, 'reverse' matches paper equations")
    parser.add_argument("--kl_topk", type=int, default=None,
                        help="If set, compute KL over student's top-k logits (teacher gathered at same indices, "
                             "subset renormalized). Typical value 100. Matches TRL v1.2 default; trades a small "
                             "approximation for large memory savings on the KL tensor.")

    # vLLM (optional, for faster on-policy generation)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use a vLLM server for on-policy generation (requires running vLLM server)")
    parser.add_argument("--vllm_port", type=int, default=8000,
                        help="Port of the vLLM OpenAI-compatible server")
    parser.add_argument("--vllm_sync_steps", type=int, default=512,
                        help="Save checkpoint for vLLM weight reload every N steps")
    parser.add_argument("--importance_sampling_correction", action="store_true", default=True,
                        help="Apply IS correction when using vLLM (default: True)")
    parser.add_argument("--no_importance_sampling_correction", dest="importance_sampling_correction",
                        action="store_false",
                        help="Disable IS correction")
    parser.add_argument("--importance_sampling_cap", type=float, default=2.0,
                        help="Cap per-sequence IS weights to prevent instability")

    # Training
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true",
                        help="Enable DDP find_unused_parameters (default: off)")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        choices=["adamw_torch", "adamw_torch_fused",
                                 "adamw_bnb_8bit", "paged_adamw_8bit",
                                 "paged_adamw_32bit"],
                        help="Optimizer. Use 'paged_adamw_8bit' (bitsandbytes) "
                             "to ~halve optimizer-state memory and fit larger "
                             "models under DDP.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup",
                                 "inverse_sqrt", "reduce_lr_on_plateau",
                                 "warmup_stable_decay"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="HF model to borrow chat template from (e.g. instruct variant for a base model)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
