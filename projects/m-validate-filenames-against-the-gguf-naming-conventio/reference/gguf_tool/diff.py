def diff_metadata(meta_hub: dict, meta_local: dict) -> dict:
    diffs = {}
    all_keys = set(meta_hub.keys()).union(set(meta_local.keys()))
    for k in all_keys:
        val_hub = meta_hub.get(k)
        val_local = meta_local.get(k)
        if val_hub != val_local:
            diffs[k] = {"hub": val_hub, "local": val_local}
    return diffs
