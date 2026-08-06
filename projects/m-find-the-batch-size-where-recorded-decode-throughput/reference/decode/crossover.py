def find_crossover_batch_size(profile_data, peak_bw, peak_flops):
    ridge_intensity = peak_flops / peak_bw
    best_bs = 1
    min_diff = float("inf")
    for row in profile_data:
        bs = row["batch_size"]
        intensity = row["intensity"]
        diff = abs(intensity - ridge_intensity)
        if diff < min_diff:
            min_diff = diff
            best_bs = bs
    return best_bs
