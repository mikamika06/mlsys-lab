import numpy as np

def argmin_index(arr):
    return int(np.argmin(arr))

def find_throughput_knee(ngls, throughputs):
    ngls = np.array(ngls)
    tps = np.array(throughputs)
    if len(ngls) == 0:
        return 0
    x = (ngls - ngls[0]) / (ngls[-1] - ngls[0] + 1e-9)
    y = (tps - tps[0]) / (tps[-1] - tps[0] + 1e-9)
    d = np.abs(x - y) / np.sqrt(2)
    idx = argmin_index(-d)
    return int(ngls[idx])
