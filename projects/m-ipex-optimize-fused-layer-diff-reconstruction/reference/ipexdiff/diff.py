def reconstruct_fused_diff(original_model, optimized_model):
    """
    Compares named modules between an original model and an optimized model.
    Returns a dict mapping module path string to a dict with 'original' and 'optimized' class names.
    """
    orig_modules = dict(original_model.named_modules())
    opt_modules = dict(optimized_model.named_modules())
    diff = {}
    for path, orig_mod in orig_modules.items():
        if path in opt_modules:
            opt_mod = opt_modules[path]
            orig_name = type(orig_mod).__name__
            opt_name = type(opt_mod).__name__
            if orig_name != opt_name:
                diff[path] = {"original": orig_name, "optimized": opt_name}
    return diff


def categorize_replacements(diff_map):
    """
    Categorizes replacement mappings into counts of each (original, optimized) pair type.
    """
    counts = {}
    for entry in diff_map.values():
        key = (entry["original"], entry["optimized"])
        counts[key] = counts.get(key, 0) + 1
    return counts
