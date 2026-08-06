import numpy as np

def measure_ep_vs_tp_throughput(ep_latencies, tp_latencies):
    ep_arr = np.array(ep_latencies, dtype=float)
    tp_arr = np.array(tp_latencies, dtype=float)
    return float(np.mean(tp_arr / ep_arr))
