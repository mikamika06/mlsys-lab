def choose_model(memory_cap_gb, model_options):
    valid = []
    for opt in model_options:
        if opt["memory_gb"] <= memory_cap_gb:
            valid.append(opt)
    if not valid:
        return None
    best = max(valid, key=lambda x: x["score"])
    return best["name"]
