import ref


def check(workdir):
    from compressor.tradeoff import compute_throughput_ratio

    ratios = []
    for cfg in ref.CONFIGS:
        want = ref.compute_throughput_ratio(cfg)
        got = compute_throughput_ratio(cfg)
        if got is not None:
            ratios.append(got / want if want != 0 else 1.0)
    avg_ratio = float(sum(ratios) / len(ratios)) if ratios else 0.0
    return {"throughput_ratio": avg_ratio}
