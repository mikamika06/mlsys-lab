TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
    "q4_1": 0.625,
}


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
