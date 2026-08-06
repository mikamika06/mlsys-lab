from snaptool.timeline import build_timeline


def compare_footprint(snapshot):
    spec = snapshot["model_spec"]
    params = spec["param_count"]
    bpp = spec["bytes_per_param"]
    opt_mult = spec["optimizer_multiplier"]
    
    theoretical = int(params * bpp * (1.0 + opt_mult))
    _, peak = build_timeline(snapshot)
    overhead = peak - theoretical
    return theoretical, overhead
