def diff_metadata(hub_meta, local_meta):
    diffs = {}
    for k in set(list(hub_meta.keys()) + list(local_meta.keys())):
        if hub_meta.get(k) != local_meta.get(k):
            diffs[k] = {"hub": hub_meta.get(k), "local": local_meta.get(k)}
    return diffs
