def map_target_to_config(target: str) -> dict:
    if target == "edge_device":
        return {"method": "torchao_int4", "group_size": 32, "asymmetric": True}
    elif target == "fine_tuning":
        return {"method": "bnb_nf4", "group_size": 64, "asymmetric": False}
    elif target == "server_inference":
        return {"method": "gptq_w4a16", "group_size": 128, "asymmetric": True}
    raise ValueError(f"Unknown target {target}")
