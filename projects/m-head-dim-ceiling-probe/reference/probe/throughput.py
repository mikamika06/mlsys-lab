def estimate_throughput(cfg):
    hd = cfg["head_dim"]
    hw = cfg["hardware"]
    if hw != "hopper":
        return {"fa2": 50.0, "fa3": 40.0}
    if hd <= 128:
        return {"fa2": 220.0, "fa3": 310.0}
    elif hd <= 256:
        return {"fa2": 150.0, "fa3": 110.0}
    else:
        return {"fa2": 80.0, "fa3": 60.0}
