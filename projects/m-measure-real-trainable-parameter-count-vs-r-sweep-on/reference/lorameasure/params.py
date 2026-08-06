def count_trainable_params(model_structure, target_modules, r, use_rslora=False):
    total = 0
    modules = model_structure.get("modules", {})
    targets = set(target_modules) if isinstance(target_modules, (list, tuple, set)) else {target_modules}
    for name, spec in modules.items():
        if spec.get("type") != "linear":
            continue
        if name in targets or any(name.endswith("." + t) for t in targets):
            in_dim = spec["in_features"]
            out_dim = spec["out_features"]
            num_params = r * (in_dim + out_dim)
            if spec.get("bias", False):
                num_params += 0
            total += num_params
    return total


def sweep_ranks(model_structure, target_modules, ranks, use_rslora=False):
    res = {}
    for r in ranks:
        res[r] = count_trainable_params(model_structure, target_modules, r, use_rslora=use_rslora)
    return res
