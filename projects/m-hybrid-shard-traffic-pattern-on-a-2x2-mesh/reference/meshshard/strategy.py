def select_strategy(strategies, memory_budget, model_memory):
    valid = []
    for s in strategies:
        req = model_memory.get(s, float("inf"))
        if req <= memory_budget:
            valid.append((s, req))
    if not valid:
        return min(strategies, key=lambda x: model_memory.get(x, float("inf")))
    return min(valid, key=lambda x: x[1])[0]
