def compute_census(nodes):
    delegated = 0
    fallback = 0
    for n in nodes:
        if n.get("target") == "xnnpack":
            delegated += 1
        else:
            fallback += 1
    total = delegated + fallback
    ratio = (delegated / total) if total > 0 else 0.0
    return {"delegated": delegated, "fallback": fallback, "ratio": ratio}
