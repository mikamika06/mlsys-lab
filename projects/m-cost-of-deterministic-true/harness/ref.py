import numpy as np

CONFIGS = [
    {"seq_len": 2048, "batch_size": 2, "heads": 32, "head_dim": 128},
    {"seq_len": 4096, "batch_size": 1, "heads": 16, "head_dim": 64},
    {"seq_len": 1024, "batch_size": 4, "heads": 8, "head_dim": 128},
    {"seq_len": 8192, "batch_size": 1, "heads": 32, "head_dim": 128}
]

def get_latency_ratio(config):
    from detcost.metrics import compute_latency_ratio as ref_lat
    return ref_lat(config)

def get_diagnose_result(losses, norms):
    from detcost.diagnose import diagnose_divergence as ref_diag
    return ref_diag(losses, norms)

def get_tradeoff_result(layers, budget, cost_fn):
    from detcost.tradeoff import optimize_checkpointing as ref_opt
    return ref_opt(layers, budget, cost_fn)
