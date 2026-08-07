import random


def generate_snapshots():
    random.seed(42)
    models = ["model_a", "model_b", "model_c", "model_d"]
    snapshots = []
    active = list(models)
    snapshots.append(list(active))
    while len(active) > 1:
        active.pop(random.randrange(len(active)))
        snapshots.append(list(active))
    return snapshots, models


def reconstruct_eviction_order(snapshots):
    order = []
    seen = set()
    flat_snapshots = [set(s) for s in snapshots]
    for i in range(len(flat_snapshots) - 1):
        current = flat_snapshots[i]
        nxt = flat_snapshots[i + 1]
        diff = current - nxt
        for m in diff:
            if m not in seen:
                seen.add(m)
                order.append(m)
    last_snap = flat_snapshots[-1]
    for m in flat_snapshots[0]:
        if m not in seen and m not in last_snap:
            seen.add(m)
            order.append(m)
    return order


def simulate_unload(model_id, api_ps_state):
    return [m for m in api_ps_state if m.get("id") != model_id]


def verify_unload(api_ps_before, api_ps_after, model_id):
    before_has = any(m.get("id") == model_id for m in api_ps_before)
    after_has = any(m.get("id") == model_id for m in api_ps_after)
    return before_has and not after_has
