CONFIGS = [
    {"seq_len": 4096, "hidden_size": 2048, "tp_size": 8, "batch_size": 2},
    {"seq_len": 8192, "hidden_size": 4096, "tp_size": 4, "batch_size": 1},
    {"seq_len": 2048, "hidden_size": 1024, "tp_size": 2, "batch_size": 4},
]

def build_region(cfg):
    return {
        "sharded_dim": 0,
        "tp_size": cfg["tp_size"],
        "seq_len": cfg["seq_len"],
        "hidden_size": cfg["hidden_size"],
        "status": "active"
    }

def measure_memory(cfg, mode):
    s = cfg["seq_len"]
    h = cfg["hidden_size"]
    b = cfg["batch_size"]
    tp = cfg["tp_size"]
    base = s * h * b * 4
    if mode == "tp_only":
        return base
    elif mode == "tp_sp":
        return base // tp
    return base

def validate_fix(sequence):
    ops = list(sequence)
    if "scatter" in ops and "gather" in ops:
        if ops.index("scatter") < ops.index("gather"):
            return True
    return False
