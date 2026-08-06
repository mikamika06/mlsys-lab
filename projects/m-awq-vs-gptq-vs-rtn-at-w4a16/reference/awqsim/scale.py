import numpy as np


def derive_awq_mappings(architecture_config):
    mapping = {}
    items = architecture_config.get(
        "layers",
        architecture_config.get("nodes", architecture_config.get("modules", [])),
    )
    for item in items:
        src = item.get("input", item.get("source", item.get("in")))
        targets = item.get("projections", item.get("branches", item.get("targets")))
        if src and targets:
            mapping[src] = sorted(list(targets))
    return mapping


def compute_awq_scales(W_list, X, alpha=0.5, max_scale_ratio=5.0):
    Sx = np.mean(np.abs(X), axis=0)
    Sw_list = [np.mean(np.abs(W), axis=0) for W in W_list]
    Sw = np.mean(Sw_list, axis=0)
    s_raw = (Sx**alpha) / (Sw ** (1.0 - alpha) + 1e-8)
    s_norm = s_raw / np.mean(s_raw)
    s_cap = max_scale_ratio * np.median(s_norm)
    s_floor = 1.0 / max_scale_ratio
    return np.clip(s_norm, s_floor, s_cap)
