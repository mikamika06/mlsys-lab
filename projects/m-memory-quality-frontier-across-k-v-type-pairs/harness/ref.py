TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
    "q4_1": 0.625,
}

SAMPLE_MODELS = [
    {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
    {"n_layers": 40, "n_kv_heads": 8, "head_dim": 128},
    {"n_layers": 28, "n_kv_heads": 4, "head_dim": 64},
]

SAMPLE_CANDIDATES = [
    {"k_type": "f16", "v_type": "f16", "perplexity_delta": 0.00},
    {"k_type": "q8_0", "v_type": "q8_0", "perplexity_delta": 0.02},
    {"k_type": "q8_0", "v_type": "q4_0", "perplexity_delta": 0.08},
    {"k_type": "q4_0", "v_type": "q4_0", "perplexity_delta": 0.15},
    {"k_type": "f32", "v_type": "f32", "perplexity_delta": 0.01},
    {"k_type": "f16", "v_type": "q8_0", "perplexity_delta": 0.03},
]

SAMPLE_FALLBACK_CASES = [
    {"k_type": "q8_0", "v_type": "q8_0", "head_dim": 128},
    {"k_type": "q8_0", "v_type": "q4_0", "head_dim": 128},
    {"k_type": "f16", "v_type": "f16", "head_dim": 64},
    {"k_type": "f16", "v_type": "f16", "head_dim": 80},
    {"k_type": "q4_1", "v_type": "q4_1", "head_dim": 128},
]

SAMPLE_CAPACITY_CASES = [
    {"budget": 12 * 1024**3, "base": 6 * 1024**3, "cfg": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128}, "k": "q8_0", "v": "q8_0"},
    {"budget": 12 * 1024**3, "base": 6 * 1024**3, "cfg": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128}, "k": "q8_0", "v": "q4_0"},
    {"budget": 16 * 1024**3, "base": 8 * 1024**3, "cfg": {"n_layers": 40, "n_kv_heads": 8, "head_dim": 128}, "k": "f16", "v": "f16"},
]


def compute_pareto_frontier(model_cfg, sequence_length, candidates):
    elements_per_token = model_cfg["n_layers"] * model_cfg["n_kv_heads"] * model_cfg["head_dim"]
    augmented = []
    for c in candidates:
        k_b = sequence_length * elements_per_token * TYPE_BYTES[c["k_type"]]
        v_b = sequence_length * elements_per_token * TYPE_BYTES[c["v_type"]]
        tot_bytes = int(round(k_b + v_b))
        item = dict(c)
        item["total_bytes"] = tot_bytes
        augmented.append(item)

    frontier = []
    for cand in augmented:
        dominated = False
        for other in augmented:
            if (
                other["total_bytes"] <= cand["total_bytes"]
                and other["perplexity_delta"] <= cand["perplexity_delta"]
                and (
                    other["total_bytes"] < cand["total_bytes"]
                    or other["perplexity_delta"] < cand["perplexity_delta"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(cand)

    frontier.sort(key=lambda x: (x["total_bytes"], x["perplexity_delta"]))
    return frontier


def detect_fa_fallback(k_type, v_type, head_dim):
    supported = {"f16", "q8_0", "q4_0"}
    if k_type not in supported or v_type not in supported:
        return {"fallback": True, "reason": "unsupported_quant_type"}
    if k_type != v_type:
        return {"fallback": True, "reason": "mismatch_kv_types"}
    if head_dim <= 0 or head_dim > 256 or head_dim % 32 != 0:
        return {"fallback": True, "reason": "unaligned_head_dim"}
    return {"fallback": False, "reason": ""}


def max_context_length(vram_budget_bytes, base_model_bytes, model_cfg, k_type="q8_0", v_type="q8_0"):
    avail = vram_budget_bytes - base_model_bytes
    if avail <= 0:
        return 0
    fb = detect_fa_fallback(k_type, v_type, model_cfg["head_dim"])
    elements_per_token = model_cfg["n_layers"] * model_cfg["n_kv_heads"] * model_cfg["head_dim"]
    unit_bytes = elements_per_token * (TYPE_BYTES[k_type] + TYPE_BYTES[v_type])
    if fb["fallback"]:
        unit_bytes *= 1.20
    return int(avail // unit_bytes)
