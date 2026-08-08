CONTEXT_CONFIGS = [
    {
        "model_max_ctx": 4096,
        "requested_c": 0,
        "requested_np": 4,
        "min_tokens": 512,
    },
    {
        "model_max_ctx": 8192,
        "requested_c": 4096,
        "requested_np": 2,
        "min_tokens": 1024,
    },
    {
        "model_max_ctx": 16384,
        "requested_c": 16384,
        "requested_np": 8,
        "min_tokens": 2048,
    },
    {
        "model_max_ctx": 32768,
        "requested_c": 8192,
        "requested_np": 16,
        "min_tokens": 1000,
    },
    {
        "model_max_ctx": 4096,
        "requested_c": 2048,
        "requested_np": 8,
        "min_tokens": 512,
    },
]

SATURATION_CONFIGS = [
    {"total_ctx": 16384, "req_slot_ctx": 2048, "gpu_slot_cap": 8},
    {"total_ctx": 8192, "req_slot_ctx": 1024, "gpu_slot_cap": 16},
    {"total_ctx": 32768, "req_slot_ctx": 4096, "gpu_slot_cap": 4},
    {"total_ctx": 12000, "req_slot_ctx": 2000, "gpu_slot_cap": 10},
    {"total_ctx": 4096, "req_slot_ctx": 8192, "gpu_slot_cap": 4},
]


def compute_slot_context(
    ctx_size: int, n_parallel: int, model_max_ctx: int = 4096
) -> int:
    """Reference implementation for slot context."""
    if n_parallel <= 0:
        return 0
    effective_ctx = model_max_ctx if ctx_size == 0 else ctx_size
    return effective_ctx // n_parallel


def plan_slot_allocation(
    model_max_ctx: int, requested_c: int, requested_np: int, min_tokens_per_slot: int
) -> dict:
    """Reference implementation for slot allocation plan."""
    effective_c = model_max_ctx if requested_c == 0 else requested_c
    if requested_np <= 0:
        return {
            "total_ctx": effective_c,
            "n_parallel": requested_np,
            "slot_ctx": 0,
            "is_valid": False,
        }
    slot_ctx = effective_c // requested_np
    is_valid = (
        (requested_np > 0)
        and (slot_ctx >= min_tokens_per_slot)
        and (effective_c <= model_max_ctx)
    )
    return {
        "total_ctx": effective_c,
        "n_parallel": requested_np,
        "slot_ctx": slot_ctx,
        "is_valid": is_valid,
    }


def find_np_saturation(
    total_ctx: int, req_slot_ctx: int, gpu_slot_cap: int = 64
) -> dict:
    """Reference implementation for np saturation point."""
    if req_slot_ctx <= 0 or total_ctx <= 0:
        return {
            "sat_np": 0,
            "slot_ctx": 0,
            "wasted_ctx": total_ctx,
            "is_saturated": True,
            "max_np_by_ctx": 0,
        }
    max_np_by_ctx = total_ctx // req_slot_ctx
    sat_np = min(max_np_by_ctx, gpu_slot_cap)
    if sat_np < 1:
        return {
            "sat_np": 0,
            "slot_ctx": 0,
            "wasted_ctx": total_ctx,
            "is_saturated": True,
            "max_np_by_ctx": max_np_by_ctx,
        }
    slot_ctx = total_ctx // sat_np
    wasted_ctx = total_ctx - (slot_ctx * sat_np)
    is_saturated = (sat_np >= gpu_slot_cap) or (sat_np == max_np_by_ctx)
    return {
        "sat_np": sat_np,
        "slot_ctx": slot_ctx,
        "wasted_ctx": wasted_ctx,
        "is_saturated": is_saturated,
        "max_np_by_ctx": max_np_by_ctx,
    }


def parse_prometheus_metrics(raw_text: str) -> dict:
    """Reference implementation for metrics parsing."""
    out = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric_expr, val_str = parts[0], parts[1]
        metric_name = metric_expr.split("{")[0]
        try:
            val = float(val_str)
            out[metric_name] = out.get(metric_name, 0.0) + val
        except ValueError:
            continue
    return out


def analyze_prompt_cache(raw_text: str) -> dict:
    """Reference implementation for prompt cache analysis."""
    metrics = parse_prometheus_metrics(raw_text)
    total_prompt = metrics.get("llamacpp:prompt_tokens_total", 0.0)
    eval_prompt = metrics.get("llamacpp:prompt_tokens_processed", 0.0)
    if "llamacpp:prompt_tokens_cached" in metrics:
        cached_prompt = metrics["llamacpp:prompt_tokens_cached"]
    else:
        cached_prompt = max(0.0, total_prompt - eval_prompt)
    hit_ratio = (cached_prompt / total_prompt) if total_prompt > 0 else 0.0
    return {
        "prompt_tokens_total": int(total_prompt),
        "prompt_tokens_processed": int(eval_prompt),
        "prompt_tokens_cached": int(cached_prompt),
        "hit_ratio": float(hit_ratio),
        "is_reusing": hit_ratio > 0.0,
    }
