import numpy as np
from autotp.latency import decode_latency

def optimal_tp_degree(config):
    lats = [decode_latency(config, tp) for tp in config["tp_degrees"]]
    idx = int(np.argmin(lats))
    return config["tp_degrees"][idx]
