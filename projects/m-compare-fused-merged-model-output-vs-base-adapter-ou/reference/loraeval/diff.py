def diff_adapter_configs(cfg1, cfg2):
    keys = set(cfg1.keys()).union(set(cfg2.keys()))
    diffs = {}
    for k in sorted(keys):
        v1 = cfg1.get(k)
        v2 = cfg2.get(k)
        if v1 != v2:
            diffs[k] = {"config1": v1, "config2": v2}
    return diffs
