import random

def generate_fixtures():
    random.seed(42)
    tables = []
    for _ in range(5):
        t = []
        for bs in [1, 2, 4, 8, 16, 32]:
            lat = float(bs * 5 + random.uniform(0, 5))
            th = float(bs * 10 - random.uniform(0, 2))
            t.append({"batch_size": bs, "latency": lat, "throughput": th})
        tables.append(t)
    return tables
