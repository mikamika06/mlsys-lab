def generate_trace():
    base = [10, 20, 30, 40]
    trace = []
    for i in range(20):
        prefix = base[:(i % 3) + 1]
        prompt = prefix + [100 + i, 200 + i]
        trace.append(prompt)
    return trace
