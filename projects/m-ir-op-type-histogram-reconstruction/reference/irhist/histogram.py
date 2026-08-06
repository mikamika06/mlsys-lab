def build_histogram(ops):
    hist = {}
    for op in ops:
        t = op["type"]
        hist[t] = hist.get(t, 0) + 1
    return dict(sorted(hist.items()))
