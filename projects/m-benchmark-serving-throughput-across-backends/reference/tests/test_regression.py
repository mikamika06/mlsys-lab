def test_backend_selection_and_performance():
    """Verify backend selection invariant and latency thresholds."""
    from serving.backend import select_backend
    from serving.benchmark import run_benchmark_pass

    env = {}
    selected = select_backend("NVIDIA H100 80GB HBM3", (9, 0), True, env)
    if selected != "FLASH_ATTN":
        raise AssertionError(f"Expected FLASH_ATTN on Hopper, got {selected}")

    pass_fa2 = run_benchmark_pass("FLASH_ATTN", 32, 512, 128)
    pass_xformers = run_benchmark_pass("XFORMERS", 32, 512, 128)

    if pass_fa2["throughput_tok_s"] <= pass_xformers["throughput_tok_s"]:
        raise AssertionError("FLASH_ATTN should achieve higher throughput than XFORMERS")

    if pass_fa2["ttft_ms"] >= pass_xformers["ttft_ms"]:
        raise AssertionError("FLASH_ATTN should achieve lower TTFT than XFORMERS")
