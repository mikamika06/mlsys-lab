def expected_adapter_shape(in_features, out_features, rank):
    return [(out_features, rank), (rank, in_features)]


def expected_peak_rss(base_mb, is_qlora):
    if is_qlora:
        return base_mb * 0.65
    return base_mb * 1.25


def expected_param_count(target_modules, rank, in_features, out_features):
    total = 0
    for _ in target_modules:
        total += rank * (in_features + out_features)
    return total
