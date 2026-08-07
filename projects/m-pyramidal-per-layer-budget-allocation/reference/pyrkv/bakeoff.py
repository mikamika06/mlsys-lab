def run_bakeoff(prompts: list[str], total_budget: int, strategies: list[dict]) -> dict[str, float]:
    out = {}
    for s in strategies:
        name = s["name"]
        score = 0.0
        for p in prompts:
            score += float((len(p) + len(name) + total_budget) % 11) / 10.0
        out[name] = round(score / max(1, len(prompts)), 4)
    return out
