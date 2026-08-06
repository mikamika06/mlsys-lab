import numpy as np

def calc_replicas(lam, service_rate, max_queue):
    effective_rate = lam / service_rate
    req_rep = np.ceil(effective_rate + np.sqrt(max_queue))
    return int(max(1, req_rep))
