import numpy as np

def find_losing_batch(table):
    bs = np.array([row["batch_size"] for row in table], dtype=float)
    lat = np.array([row["latency"] for row in table], dtype=float)
    throughput = bs / lat
    peak = int(np.argmax(throughput))
    drop_idx = peak + int(np.argmin(throughput[peak:]))
    return int(bs[drop_idx])
