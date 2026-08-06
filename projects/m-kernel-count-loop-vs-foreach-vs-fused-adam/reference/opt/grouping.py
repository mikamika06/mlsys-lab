def group_tensors_by_device_dtype(params):
    groups = {}
    for p in params:
        key = (p.get("device", "cpu"), p.get("dtype", "float32"))
        if key not in groups:
            groups[key] = []
        groups[key].append(p)
    return groups


def estimate_kernel_counts(params, num_steps=1):
    n_params = len(params)
    groups = group_tensors_by_device_dtype(params)
    n_groups = len(groups)

    return {
        "loop": n_params * 4 * num_steps,
        "foreach": n_groups * 4 * num_steps,
        "fused": n_groups * 1 * num_steps,
    }
