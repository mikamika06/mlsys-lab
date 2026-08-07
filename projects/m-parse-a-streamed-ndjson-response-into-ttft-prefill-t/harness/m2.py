import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from streammetrics.bench import compute_median_decode_throughput

    dataset = ref.generate_dataset(num_traces=8, seed=456)
    streams = [item["stream"] for item in dataset]

    want_warmup_1 = ref.ref_compute_median_decode_throughput(streams, warmup_runs=1)
    want_warmup_2 = ref.ref_compute_median_decode_throughput(streams, warmup_runs=2)

    try:
        got_warmup_1 = compute_median_decode_throughput(streams, warmup_runs=1)
        got_warmup_2 = compute_median_decode_throughput(streams, warmup_runs=2)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Exception during benchmarking: {e}"}

    err1 = abs(got_warmup_1 - want_warmup_1) / max(1e-9, abs(want_warmup_1))
    err2 = abs(got_warmup_2 - want_warmup_2) / max(1e-9, abs(want_warmup_2))

    return {"rel_err": max(err1, err2)}
