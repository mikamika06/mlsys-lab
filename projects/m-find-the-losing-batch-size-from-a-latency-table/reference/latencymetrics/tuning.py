def tune_max_seqs(lat_model_fn, ttft_slo, max_range):
    best = max_range[0]
    for m in max_range:
        lat = lat_model_fn(m)
        if lat <= ttft_slo:
            best = m
        else:
            break
    return best
