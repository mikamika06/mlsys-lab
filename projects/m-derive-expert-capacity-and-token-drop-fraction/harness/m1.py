import ref


def check(workdir):
    from moe.capacity import compute_expert_capacity

    out = {"capacity_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_expert_capacity(
            cfg["num_tokens"], cfg["num_experts"], cfg["capacity_factor"], cfg["top_k"]
        )
        got = compute_expert_capacity(
            cfg["num_tokens"], cfg["num_experts"], cfg["capacity_factor"], cfg["top_k"]
        )
        if got == want:
            ok += 1
    out["capacity_matched"] = float(ok)
    return out
