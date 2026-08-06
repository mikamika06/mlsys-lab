import ref


def check(workdir):
    from flexmask.cost import BlockMaskCostProfiler

    out = {"cost_analysis_correct": 0.0, "latency_ratio": 0.0}
    correct_count = 0
    total_configs = len(ref.TEST_CONFIGS)
    max_latency_ratio = 0.0

    for cfg in ref.TEST_CONFIGS:
        want = ref.run_profiler_reference(cfg)
        profiler = BlockMaskCostProfiler(block_size=cfg["block_size"])

        got_ops = profiler.compute_dense_mask_ops(cfg["seq_len"], cfg["seq_len"])
        got_blocks = profiler.compute_blockmask_sparse_blocks(
            cfg["seq_len"], cfg["seq_len"], "causal"
        )
        got_sim = profiler.simulate_flex_vs_fa2_latency(
            seq_len=cfg["seq_len"], num_heads=cfg["heads"]
        )

        if (
            got_ops == want["dense_ops"]
            and got_blocks == want["blocks"]
            and abs(got_sim["latency_ratio"] - want["sim"]["latency_ratio"]) < 1e-4
        ):
            correct_count += 1

        if got_sim["latency_ratio"] > max_latency_ratio:
            max_latency_ratio = got_sim["latency_ratio"]

    if correct_count == total_configs:
        out["cost_analysis_correct"] = 1.0

    out["latency_ratio"] = float(max_latency_ratio)
    return out
