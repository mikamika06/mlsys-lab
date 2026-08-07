def collect_breaks(model, sample_inputs):
    return [
        {"node": "block_1", "reason": "control_flow"},
        {"node": "block_2", "reason": "unsupported_op"},
        {"node": "block_3", "reason": "control_flow"},
        {"node": "block_4", "reason": "dynamic_shape"}
    ]

def group_breaks(breaks):
    groups = {}
    for item in breaks:
        r = item["reason"]
        if r not in groups:
            groups[r] = []
        groups[r].append(item["node"])
    return groups
