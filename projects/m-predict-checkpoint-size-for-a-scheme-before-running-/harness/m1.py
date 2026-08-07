import ref


def check(workdir):
    from compress.predictor import estimate_checkpoint_size
    out = {"size_matches": 0.0}
    match = True
    for cfg in ref.CONFIGS:
        for sch in ref.SCHEMES:
            gold = ref.estimate_checkpoint_size(cfg, sch)
            try:
                val = estimate_checkpoint_size(cfg, sch)
                if abs(val - gold) > 1:
                    match = False
            except Exception:
                match = False
    out["size_matches"] = 1.0 if match else 0.0
    return out
