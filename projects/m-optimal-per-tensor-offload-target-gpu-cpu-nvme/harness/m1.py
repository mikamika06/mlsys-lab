import ref


def check(workdir):
    from offload_target.placement import select_offload_targets

    out = {"placements_matched": 0.0, "valid_capacity_count": 0.0}
    matched = 0
    valid_cap = 0

    for i in range(len(ref.TENSOR_SETS)):
        tensors = ref.TENSOR_SETS[i]
        hw = ref.HARDWARE_CONFIGS[i]
        want = ref.select_offload_targets(tensors, hw)
        got = select_offload_targets(tensors, hw)

        if got == want:
            matched += 1

        usage = got.get("device_usage", {})
        assignments = got.get("assignments", {})
        calc_usage = {0: 0, 1: 0, 2: 0}
        for t in tensors:
            tid = t["id"]
            if tid in assignments:
                calc_usage[assignments[tid]] += t["size_bytes"]

        capacities_ok = all(
            calc_usage[d] <= hw[d]["capacity_bytes"] for d in (0, 1, 2)
        )
        if capacities_ok and usage == calc_usage:
            valid_cap += 1

    out["placements_matched"] = float(matched)
    out["valid_capacity_count"] = float(valid_cap)
    return out
