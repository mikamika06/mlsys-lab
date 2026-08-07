def tensor_cost(tensor, target_spec):
    bw_bytes_per_sec = target_spec["bandwidth_gbps"] * 1e9
    latency_sec = target_spec["latency_us"] * 1e-6
    transfer_sec = tensor["size_bytes"] / bw_bytes_per_sec
    return (transfer_sec + latency_sec) * tensor["access_frequency"]


def select_offload_targets(tensors, hardware):
    sorted_tensors = sorted(
        tensors,
        key=lambda t: (t["access_frequency"] / t["size_bytes"], t["id"]),
        reverse=True,
    )
    used_bytes = {0: 0, 1: 0, 2: 0}
    placements = {}

    for t in sorted_tensors:
        best_target = None
        best_cost = float("inf")
        for dev in (0, 1, 2):
            if used_bytes[dev] + t["size_bytes"] <= hardware[dev]["capacity_bytes"]:
                c = tensor_cost(t, hardware[dev])
                if c < best_cost:
                    best_cost = c
                    best_target = dev
        if best_target is None:
            best_target = 2
        used_bytes[best_target] += t["size_bytes"]
        placements[t["id"]] = {
            "target": best_target,
            "cost": best_cost,
        }

    return {
        "assignments": {tid: data["target"] for tid, data in placements.items()},
        "device_usage": used_bytes,
    }
