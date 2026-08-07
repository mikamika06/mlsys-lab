def find_imbalanced_stage(logs):
    stage_allocations = {}
    for entry in logs:
        stage = entry["stage"]
        mem = entry["activation_bytes"]
        stage_allocations.setdefault(stage, []).append(mem)

    means = {s: sum(m) / len(m) for s, m in stage_allocations.items()}
    max_stage = max(means, key=means.get)
    return int(max_stage)
