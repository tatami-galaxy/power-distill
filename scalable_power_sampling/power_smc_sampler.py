"""HuggingFace-backed Power-SMC sampler.

This implements the sequence-level Power-SMC algorithm from
"Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM
Reasoning" for use in this repository's evaluation scripts.

Unlike the vLLM rollout sampler in :mod:`vllm_batched_power_sampler`, this
sampler needs direct access to ``past_key_values`` so it can reorder KV caches
after SMC resampling. It therefore uses Transformers/PyTorch directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _torch_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    if dtype == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _logsumexp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim)


def _alpha_ramp(t: int, alpha_final: float, ramp_tokens: int) -> float:
    if ramp_tokens <= 1:
        return float(alpha_final)
    if t < ramp_tokens:
        frac = float(t + 1) / float(ramp_tokens)
        return 1.0 + (float(alpha_final) - 1.0) * frac
    return float(alpha_final)


def _effective_sample_size(w: torch.Tensor, eps: float = 1e-12) -> float:
    w = w.clamp_min(eps)
    return float(1.0 / torch.sum(w * w).item())


def _systematic_resample(w: torch.Tensor, generator=None) -> torch.Tensor:
    n = w.numel()
    device = w.device
    if generator is None:
        u0 = torch.rand((), device=device)
    else:
        u0 = torch.rand((), device=device, generator=generator)
    positions = (u0 + torch.arange(n, device=device)) / n
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    idx = torch.searchsorted(cdf, positions, right=False)
    return idx.clamp_max(n - 1).to(torch.long)


def _top_k_filter_(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
    logits.masked_fill_(logits < kth_vals, -float("inf"))
    return logits


def _top_p_filter_(
    logits: torch.Tensor,
    top_p: float | None,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumprobs > top_p
    if min_tokens_to_keep > 1:
        sorted_mask[:, :min_tokens_to_keep] = False
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
    logits.masked_fill_(mask, -float("inf"))
    return logits


def _apply_repetition_penalty_(
    logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    penalty: float | None,
) -> torch.Tensor:
    if penalty is None or penalty == 1.0:
        return logits
    n, _ = logits.shape
    for i in range(n):
        toks = prev_tokens[i]
        if toks.numel() == 0:
            continue
        uniq = torch.unique(toks)
        row = logits[i]
        vals = row.index_select(0, uniq)
        vals = torch.where(vals < 0, vals * penalty, vals / penalty)
        row.index_copy_(0, uniq, vals)
    return logits


def _recursive_reorder(obj: Any, idx: torch.Tensor) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if obj.dim() >= 1 and obj.size(0) == idx.numel():
            return obj.index_select(0, idx)
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_reorder(x, idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_reorder(x, idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_reorder(v, idx) for k, v in obj.items()}
    return obj


def _recursive_select_batch(obj: Any, idx: torch.Tensor) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if idx.numel() == 0:
            return obj
        if obj.dim() >= 1 and obj.size(0) >= idx.max().item() + 1:
            return obj.index_select(0, idx).contiguous()
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_select_batch(x, idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_select_batch(x, idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_select_batch(v, idx) for k, v in obj.items()}
    return obj


def _recursive_expand_batch(obj: Any, expand_idx: torch.Tensor) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if expand_idx.numel() == 0:
            return obj
        unique_batch = int(expand_idx.max().item()) + 1
        if obj.dim() >= 1 and obj.size(0) == unique_batch:
            return obj.index_select(0, expand_idx)
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_expand_batch(x, expand_idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_expand_batch(x, expand_idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_expand_batch(v, expand_idx) for k, v in obj.items()}
    return obj


def _reorder_past_key_values(model, past_key_values, idx: torch.Tensor):
    if past_key_values is None:
        return None
    if hasattr(model, "_reorder_cache") and callable(model._reorder_cache):
        try:
            return model._reorder_cache(past_key_values, idx)
        except Exception:
            pass
    if hasattr(past_key_values, "reorder_cache") and callable(
        past_key_values.reorder_cache
    ):
        try:
            past_key_values.reorder_cache(idx)
            return past_key_values
        except Exception:
            pass
    return _recursive_reorder(past_key_values, idx)


def _select_cache_subset(model, past, idx: torch.Tensor):
    if past is None:
        return None
    if hasattr(past, "reorder_cache") and callable(past.reorder_cache):
        try:
            past.reorder_cache(idx)
            return past
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(model._reorder_cache):
        try:
            return model._reorder_cache(past, idx)
        except Exception:
            pass
    return _recursive_select_batch(past, idx)


def _expand_cache(model, past, expand_idx: torch.Tensor):
    if past is None:
        return None
    if hasattr(past, "reorder_cache") and callable(past.reorder_cache):
        try:
            past.reorder_cache(expand_idx)
            return past
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(model._reorder_cache):
        try:
            return model._reorder_cache(past, expand_idx)
        except Exception:
            pass
    return _recursive_expand_batch(past, expand_idx)


def _build_prompt_cache(model, input_ids_1: torch.Tensor, n_particles: int):
    out = model(input_ids=input_ids_1, use_cache=True)
    past_1 = getattr(out, "past_key_values", None)
    logits_1 = out.logits[:, -1, :]
    expand_idx = torch.zeros(
        n_particles,
        dtype=torch.long,
        device=input_ids_1.device,
    )
    past_n = _expand_cache(model, past_1, expand_idx)
    logits_n = logits_1.expand(n_particles, -1).contiguous()
    return past_n, logits_n


_BOX_ONLY_RE = re.compile(r"^[\s{}]*$")


def _has_nonempty_boxed(text: str) -> bool:
    for macro in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(macro, start)
            if idx < 0:
                break
            j = idx + len(macro)
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                depth = 0
                right = None
                for k in range(j, len(text)):
                    c = text[k]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            right = k
                            break
                if right is not None:
                    content = text[j + 1 : right].strip()
                    while (
                        len(content) >= 2
                        and content[0] == "{"
                        and content[-1] == "}"
                    ):
                        inner = content[1:-1].strip()
                        if inner == content:
                            break
                        content = inner
                    if content and (_BOX_ONLY_RE.fullmatch(content) is None):
                        return True
            start = idx + len(macro)
    return False


@dataclass
class PowerSMCConfig:
    max_new_tokens: int = 3072
    alpha: float = 2.0
    n_particles: int = 32
    ess_threshold: float = 0.5
    proposal_temperature: float | None = None
    block_size: int = 128
    alpha_ramp_tokens: int = 400
    min_new_tokens: int = 100
    repetition_penalty: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    min_tokens_to_keep: int = 1
    penalize_prompt: bool = False
    hard_truncation: bool = True
    soft_truncation_value: float = -50.0
    force_eos_after_done: bool = True
    stop_on_boxed: bool = True
    boxed_check_window_tokens: int = 256
    use_cow_cache: bool = True
    shared_prompt_cache: bool = True


@torch.no_grad()
def smc_power_sample(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: PowerSMCConfig,
) -> dict[str, Any]:
    """Run Power-SMC for one prompt and return weighted particles."""
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape [1, prompt_len]")

    device = input_ids.device
    generator = torch.Generator(device=device)
    n_particles = int(cfg.n_particles)
    prompt_len = input_ids.size(1)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is required for Power-SMC")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    if cfg.shared_prompt_cache:
        past, last_logits = _build_prompt_cache(model, input_ids, n_particles)
    else:
        prompt = input_ids.expand(n_particles, -1).contiguous()
        out = model(input_ids=prompt, use_cache=True)
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]

    total_len = prompt_len + int(cfg.max_new_tokens)
    seqs = torch.full(
        (n_particles, total_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    seqs[:, :prompt_len] = input_ids.expand(n_particles, -1)
    cur = prompt_len

    done = torch.zeros(n_particles, dtype=torch.bool, device=device)
    cum_logp = torch.zeros(n_particles, dtype=torch.float32, device=device)
    log_w = torch.zeros(n_particles, dtype=torch.float32, device=device)

    alpha_final = float(cfg.alpha)
    ramp_tokens = max(int(cfg.alpha_ramp_tokens), 1)
    prev_alpha = 1.0
    min_new = max(int(cfg.min_new_tokens), 0)

    stats: dict[str, Any] = {
        "resample_count": 0,
        "ess_history": [],
        "mean_logw_history": [],
        "max_logw_history": [],
        "unique_ancestors_history": [],
        "peak_batch_size_history": [],
        "done_at": None,
    }

    cow_active = False
    logical_to_physical = None
    n_physical = n_particles
    proposal_temperature = (
        float(cfg.proposal_temperature)
        if cfg.proposal_temperature is not None
        else 1.0 / alpha_final
    )

    for t in range(int(cfg.max_new_tokens)):
        alpha_t = _alpha_ramp(t=t, alpha_final=alpha_final, ramp_tokens=ramp_tokens)

        base_logprobs_phys = F.log_softmax(last_logits.float(), dim=-1)
        if proposal_temperature == 1.0:
            prop_logits_phys = last_logits.float()
        else:
            prop_logits_phys = last_logits.float() / proposal_temperature

        if cow_active and logical_to_physical is not None:
            base_logprobs = base_logprobs_phys[logical_to_physical]
            prop_logits = prop_logits_phys[logical_to_physical]
        else:
            base_logprobs = base_logprobs_phys
            prop_logits = prop_logits_phys

        if t < min_new:
            prop_logits[:, eos_id] = -float("inf")

        if cfg.repetition_penalty is not None and cfg.repetition_penalty != 1.0:
            history = seqs[:, :cur] if cfg.penalize_prompt else seqs[:, prompt_len:cur]
            _apply_repetition_penalty_(prop_logits, history, cfg.repetition_penalty)

        _top_k_filter_(prop_logits, cfg.top_k)
        _top_p_filter_(prop_logits, cfg.top_p, cfg.min_tokens_to_keep)
        if not cfg.hard_truncation:
            prop_logits = torch.where(
                torch.isinf(prop_logits),
                torch.full_like(prop_logits, float(cfg.soft_truncation_value)),
                prop_logits,
            )

        prop_logprobs = F.log_softmax(prop_logits, dim=-1)
        next_tokens = torch.empty(n_particles, dtype=torch.long, device=device)
        if cfg.force_eos_after_done:
            active = ~done
            if active.any():
                probs_active = torch.exp(prop_logprobs[active])
                next_tokens[active] = torch.multinomial(
                    probs_active,
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            next_tokens[done] = eos_id
        else:
            probs = torch.exp(prop_logprobs)
            next_tokens = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)

        idx_tok = next_tokens.view(n_particles, 1)
        token_logp = torch.gather(base_logprobs, dim=-1, index=idx_tok).squeeze(-1)
        token_logq = torch.gather(prop_logprobs, dim=-1, index=idx_tok).squeeze(-1)
        if cfg.force_eos_after_done:
            token_logp = torch.where(done, torch.zeros_like(token_logp), token_logp)
            token_logq = torch.where(done, torch.zeros_like(token_logq), token_logq)

        delta = float(alpha_t - prev_alpha)
        if delta != 0.0:
            log_w = log_w + delta * cum_logp
            prev_alpha = float(alpha_t)
        log_w = log_w + (float(alpha_t) - 1.0) * token_logp
        log_w = log_w + (token_logp - token_logq)
        cum_logp = cum_logp + token_logp

        seqs[:, cur] = next_tokens
        cur += 1

        newly_done = (next_tokens == eos_id) & (~done)
        done = done | newly_done

        if cfg.stop_on_boxed:
            active = ~done
            if active.any():
                start_tok = max(prompt_len, cur - int(cfg.boxed_check_window_tokens))
                active_idx = torch.nonzero(active, as_tuple=False).flatten().tolist()
                for particle_idx in active_idx:
                    gen_text_tail = tokenizer.decode(
                        seqs[particle_idx, start_tok:cur].tolist(),
                        skip_special_tokens=True,
                    )
                    if _has_nonempty_boxed(gen_text_tail):
                        done[particle_idx] = True

        if stats["done_at"] is None and done.all():
            stats["done_at"] = t + 1
        if done.all():
            break

        if cow_active and logical_to_physical is not None:
            past = _expand_cache(model, past, logical_to_physical)
            cow_active = False
            logical_to_physical = None
            n_physical = n_particles

        out = model(
            input_ids=next_tokens.view(n_particles, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]

        is_block_end = (
            ((t + 1) % int(cfg.block_size) == 0)
            or (t == int(cfg.max_new_tokens) - 1)
        )
        if not is_block_end:
            continue

        lw = log_w - _logsumexp(log_w, dim=0)
        w = torch.exp(lw)
        ess = _effective_sample_size(w)
        stats["ess_history"].append(ess)
        stats["mean_logw_history"].append(float(log_w.mean().item()))
        stats["max_logw_history"].append(float(log_w.max().item()))

        if ess >= float(cfg.ess_threshold) * n_particles:
            continue

        idx_rs = _systematic_resample(w, generator=generator)
        if cfg.use_cow_cache:
            unique_ancestors, inverse = torch.unique(idx_rs, return_inverse=True)
            unique_count = unique_ancestors.numel()
            stats["unique_ancestors_history"].append(int(unique_count))

            seqs = seqs.index_select(0, idx_rs)
            done = done.index_select(0, idx_rs)
            cum_logp = cum_logp.index_select(0, idx_rs)

            past = _select_cache_subset(model, past, unique_ancestors)
            last_logits = last_logits.index_select(0, unique_ancestors)
            logical_to_physical = inverse
            cow_active = True
            n_physical = int(unique_count)
        else:
            seqs = seqs.index_select(0, idx_rs)
            done = done.index_select(0, idx_rs)
            cum_logp = cum_logp.index_select(0, idx_rs)
            past = _reorder_past_key_values(model, past, idx_rs)
            last_logits = last_logits.index_select(0, idx_rs)
            stats["unique_ancestors_history"].append(n_particles)

        log_w = torch.zeros_like(log_w)
        stats["resample_count"] += 1
        stats["peak_batch_size_history"].append(n_physical)

    if prev_alpha != alpha_final:
        log_w = log_w + float(alpha_final - prev_alpha) * cum_logp

    seqs_out = seqs[:, :cur]
    lw = log_w - _logsumexp(log_w, dim=0)
    w = torch.exp(lw)
    chosen_idx = int(torch.multinomial(w, 1, generator=generator).item())
    chosen_sequence = seqs_out[chosen_idx]

    return {
        "sequences": seqs_out,
        "log_w": log_w,
        "w": w,
        "chosen_idx": chosen_idx,
        "chosen_sequence": chosen_sequence,
        "stats": stats,
    }


class HFPowerSMCSampler:
    """Power-SMC sampler with a ``generate`` API matching local eval samplers."""

    def __init__(
        self,
        model_name: str,
        alpha: float = 4.0,
        n_particles: int = 64,
        ess_threshold: float = 0.5,
        proposal_temperature: float | None = None,
        block_size: int = 64,
        alpha_ramp_tokens: int = 100,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        dtype: str = "bfloat16",
        stop_on_boxed: bool = True,
        use_cow_cache: bool = True,
        shared_prompt_cache: bool = True,
    ):
        self.config = PowerSMCConfig(
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            proposal_temperature=proposal_temperature,
            block_size=block_size,
            alpha_ramp_tokens=alpha_ramp_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            stop_on_boxed=stop_on_boxed,
            use_cow_cache=use_cow_cache,
            shared_prompt_cache=shared_prompt_cache,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(dtype),
            "trust_remote_code": False,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt: str | None = None,
        input_ids: list[int] | torch.Tensor | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        if input_ids is None:
            if prompt is None:
                raise ValueError("Provide either prompt or input_ids")
            input_ids = self.tokenizer.encode(prompt)
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.view(-1).tolist()

        prompt_len = len(input_ids)
        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        result = smc_power_sample(
            self.model,
            self.tokenizer,
            input_tensor,
            self.config,
        )
        chosen_sequence = result["chosen_sequence"]
        generated_ids = chosen_sequence[prompt_len:].detach().cpu().tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if verbose:
            stats = result["stats"]
            print(
                "Power-SMC: "
                f"{len(generated_ids)} tokens, "
                f"{stats['resample_count']} resamples"
            )

        return {
            "text": text,
            "input_ids": chosen_sequence.detach().cpu().tolist(),
            "num_tokens_generated": len(generated_ids),
            "stats": result["stats"],
        }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            "HFPowerSMCSampler("
            f"alpha={cfg.alpha}, n_particles={cfg.n_particles}, "
            f"ess_threshold={cfg.ess_threshold}, block_size={cfg.block_size}, "
            f"alpha_ramp_tokens={cfg.alpha_ramp_tokens})"
        )
