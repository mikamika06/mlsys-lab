def simulate(requests: list[dict], age_factor: float) -> dict[int, int]:
    pending = {r["id"]: dict(r) for r in requests}
    t = 0
    completions = {}

    while pending:
        available = [r for r in pending.values() if r["arrival"] <= t]
        if not available:
            t += 1
            continue

        best = None
        best_score = -1e9

        for r in available:
            score = r["prio"] + (t - r["arrival"]) * age_factor
            if score > best_score or (score == best_score and (best is None or r["id"] < best["id"])):
                best = r
                best_score = score

        best["work"] -= 1
        if best["work"] == 0:
            completions[best["id"]] = t + 1
            del pending[best["id"]]

        t += 1

    return completions
