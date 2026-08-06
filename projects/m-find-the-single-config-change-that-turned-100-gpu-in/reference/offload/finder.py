def find_config_change(c1, c2):
    diffs = {}
    keys = set(c1.keys()).union(set(c2.keys()))
    for k in keys:
        if c1.get(k) != c2.get(k):
            diffs[k] = (c1.get(k), c2.get(k))
    return diffs
