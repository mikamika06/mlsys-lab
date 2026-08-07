def compare_engines(engine_a_data, engine_b_data, memory_budget_bytes):
    def score(data):
        if data["memory_bytes"] > memory_budget_bytes:
            return -1.0
        return data["throughput"] / (data["memory_bytes"] + 1.0)

    score_a = score(engine_a_data)
    score_b = score(engine_b_data)
    if score_a > score_b:
        winner = "engine_a"
    elif score_b > score_a:
        winner = "engine_b"
    else:
        winner = "tie"
    return {"winner": winner, "score_a": score_a, "score_b": score_b}
