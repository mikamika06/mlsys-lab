import ref


def check(workdir):
    from splitkv.crossover import find_crossover_batch
    from splitkv.curve import predict_latency

    out = {"crossover_matched": 1.0, "latency_ratio": 0.0}

    ratios = []

    for sl in ref.CROSSOVER_SEQS:
        want_b = ref.find_crossover_batch(sl, 108)
        got_b = find_crossover_batch(sl, 108)

        if want_b != got_b:
            out["crossover_matched"] = 0.0
            out["_note"] = f"seq_len={sl}: expected crossover batch {want_b}, got {got_b}"

        lat_1 = predict_latency(got_b, sl, 1, 108)
        lat_opt = predict_latency(got_b, sl, ref.optimal_num_splits(got_b, sl, 108), 108)
        if lat_1 > 0:
            ratios.append(lat_opt / lat_1)

    out["latency_ratio"] = max(ratios) if ratios else 1.0
    return out
