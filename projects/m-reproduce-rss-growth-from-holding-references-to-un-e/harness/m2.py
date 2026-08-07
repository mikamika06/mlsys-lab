"""Check Milestone 2: mx.compile vs torch.compile and recompilation measurement."""

import ref


def check(workdir):
    from mlxgraph.bench import measure_mlx_recompilation_cost, profile_mlx_vs_torch_mps

    out = {"compilation_measured": 0, "recompile_penalty_ratio": 0.0}

    prof = profile_mlx_vs_torch_mps(ref.GRAPH_SPEC, warmup_runs=2, active_runs=5)

    if "mx_compile" in prof and "torch_aot_eager" in prof and "speedup_ratio" in prof:
        if prof["mx_compile"]["total_ms"] > 0 and prof["torch_aot_eager"]["total_ms"] > 0:
            out["compilation_measured"] = 1

    rec = measure_mlx_recompilation_cost(ref.GRAPH_SPEC, ref.SHAPE_SEQUENCE)

    want_recompile, want_cached = ref.expected_recompile_counts(ref.SHAPE_SEQUENCE)

    got_recompile = rec.get("recompile_count", -1)
    got_cached = rec.get("cached_count", -1)

    if got_recompile == want_recompile and got_cached == want_cached:
        penalty = float(rec.get("recompile_penalty_ratio", 0.0))
        out["recompile_penalty_ratio"] = penalty
    else:
        out["_note"] = f"Recompilation count mismatch: got ({got_recompile}, {got_cached}), expected ({want_recompile}, {want_cached})"

    return out
