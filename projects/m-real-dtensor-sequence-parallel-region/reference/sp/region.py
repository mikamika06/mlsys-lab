def build_region(cfg):
    return {
        "sharded_dim": 0,
        "tp_size": cfg["tp_size"],
        "seq_len": cfg["seq_len"],
        "hidden_size": cfg["hidden_size"],
        "status": "active"
    }
