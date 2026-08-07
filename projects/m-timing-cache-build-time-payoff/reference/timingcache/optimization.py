def find_knee(levels, latencies, compile_times):
    scores = [(l, lat * ct) for l, lat, ct in zip(levels, latencies, compile_times)]
    best = min(scores, key=lambda x: x[1])
    return best[0]
