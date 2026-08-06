import numpy as np

def compute_goodput(offered_loads, ttfts, slo=2.0):
    loads = np.array(offered_loads)
    t = np.array(ttfts)
    success = t <= slo
    goodput = loads * success
    return goodput.tolist()
