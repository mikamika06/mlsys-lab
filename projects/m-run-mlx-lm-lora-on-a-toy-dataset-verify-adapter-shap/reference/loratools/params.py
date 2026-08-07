def compute_adapter_parameters(target_modules, rank, in_features, out_features):
    total = 0
    for _ in target_modules:
        total += rank * (in_features + out_features)
    return total
