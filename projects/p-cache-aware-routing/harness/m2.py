import ref


def check(workdir):
    from routing.affinity import compute_affinity
    m = {"affinity_ok": 0.0}
    state = {10, 20, 30}
    prompt = [10, 20, 50, 60]
    score = compute_affinity(prompt, state)
    if abs(score - 0.5) < 1e-5:
        m["affinity_ok"] = 1.0
    return m
