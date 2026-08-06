import numpy as np

def compute_latency_ratio(config):
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    heads = config["heads"]
    head_dim = config["head_dim"]
    base_cost = batch_size * seq_len * seq_len * heads * head_dim * 1e-7
    det_factor = 1.35 + 0.15 * np.log2(max(seq_len, 1024) / 1024.0)
    return float(det_factor)

def compute_memory_overhead(config):
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    heads = config["heads"]
    head_dim = config["head_dim"]
    nondet_mem = batch_size * seq_len * heads * head_dim * 4
    det_mem = nondet_mem * 1.25
    return float(det_mem - nondet_mem)
