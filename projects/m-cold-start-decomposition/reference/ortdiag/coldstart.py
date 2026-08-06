def decompose_cold_start(timings):
    if not timings:
        return {"init": 0.0, "graph": 0.0, "steady": 0.0}
    init_phase = timings[0] * 0.4
    graph_phase = timings[0] * 0.4
    steady_phase = sum(timings[1:]) / max(len(timings) - 1, 1) if len(timings) > 1 else timings[0] * 0.2
    return {
        "init": float(init_phase),
        "graph": float(graph_phase),
        "steady": float(steady_phase)
    }
