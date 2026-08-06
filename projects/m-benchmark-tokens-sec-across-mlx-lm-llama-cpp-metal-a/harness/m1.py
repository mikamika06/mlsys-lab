import ref


def check(workdir):
    from benchedge.runner import run_backend_benchmark

    out = {"metrics_matched": 0.0, "throughput_ratio": 0.0}
    matched = 0
    r_mlx = None
    r_torch = None

    for backend in ["mlx-lm", "llama.cpp-metal", "torch-mps"]:
        trace = ref.TRACES[backend]
        want = ref.compute_ref_metrics(backend, trace)
        got = run_backend_benchmark(
            backend,
            trace["prompt_tokens"],
            trace["generated_tokens"],
            ref.mock_trace_fn,
        )

        if (
            got.backend == want.backend
            and abs(got.ttft_sec - want.ttft_sec) < 1e-4
            and abs(got.decode_duration_sec - want.decode_duration_sec) < 1e-4
            and abs(got.decode_tokens_per_sec - want.decode_tokens_per_sec)
            < 1e-2
        ):
            matched += 1

        if backend == "mlx-lm":
            r_mlx = got
        elif backend == "torch-mps":
            r_torch = got

    out["metrics_matched"] = float(matched)
    if r_mlx and r_torch and r_torch.decode_tokens_per_sec > 0:
        out["throughput_ratio"] = float(
            r_mlx.decode_tokens_per_sec / r_torch.decode_tokens_per_sec
        )

    return out
