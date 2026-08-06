import ref


def check(workdir):
    from benchedge.runner import run_backend_benchmark
    from benchedge.summary import summarize_benchmark_runs

    out = {"rss_matched": 0.0, "summary_matched": 0.0}

    results = []
    for backend in ["mlx-lm", "llama.cpp-metal", "torch-mps"]:
        trace = ref.TRACES[backend]
        res = run_backend_benchmark(
            backend,
            trace["prompt_tokens"],
            trace["generated_tokens"],
            ref.mock_trace_fn,
        )
        results.append(res)

    rss_ok = True
    for res in results:
        want_rss = max(ref.TRACES[res.backend]["rss_samples"])
        if abs(res.peak_rss_mb - want_rss) > 1e-2:
            rss_ok = False
            break

    if rss_ok:
        out["rss_matched"] = 1.0

    got_summary = summarize_benchmark_runs(
        results, baseline_backend="torch-mps"
    )
    want_summary = ref.build_ref_summary(results, baseline_backend="torch-mps")

    if got_summary == want_summary:
        out["summary_matched"] = 1.0
    else:
        out["_note"] = f"summary mismatch: got {got_summary}, want {want_summary}"

    return out
