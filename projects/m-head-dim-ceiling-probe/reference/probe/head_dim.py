def check_head_ceiling(cfg):
    hd = cfg["head_dim"]
    if hd <= 64:
        return {"max_supported_dim": 64, "fa2_supported": True, "fa3_supported": True, "optimal_block": 64}
    elif hd <= 128:
        return {"max_supported_dim": 128, "fa2_supported": True, "fa3_supported": True, "optimal_block": 128}
    elif hd <= 256:
        return {"max_supported_dim": 256, "fa2_supported": True, "fa3_supported": False, "optimal_block": 128}
    else:
        return {"max_supported_dim": 512, "fa2_supported": False, "fa3_supported": False, "optimal_block": 256}
