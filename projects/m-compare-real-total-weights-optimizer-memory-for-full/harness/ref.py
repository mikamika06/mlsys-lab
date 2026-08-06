import random


def get_configs():
    rng = random.Random(42)
    configs = []
    for i in range(5):
        num_layers = rng.randint(2, 6)
        hidden_dim = rng.choice([512, 1024, 2048])
        lora_rank = rng.choice([4, 8, 16])
        precision_bytes = rng.choice([2, 4])
        cfg = {
            "model_id": f"model_{i}",
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "lora_rank": lora_rank,
            "precision_bytes": precision_bytes,
        }
        configs.append(cfg)
    return configs


def count_parameters(cfg):
    L = cfg["num_layers"]
    h = cfg["hidden_dim"]
    r = cfg["lora_rank"]
    weight_per_layer = 4 * h * h
    total_base = L * weight_per_layer
    lora_per_layer = 2 * h * r + 2 * r * h
    total_lora_trainable = L * lora_per_layer
    return {
        "total_base": total_base,
        "full_trainable": total_base,
        "lora_trainable": total_lora_trainable,
        "lora_frozen": total_base,
    }


def compute_optimizer_bytes(cfg, mode):
    params = count_parameters(cfg)
    p_bytes = cfg["precision_bytes"]
    if mode == "full":
        trainable = params["full_trainable"]
        opt_bytes = 2 * trainable * 4 + trainable * p_bytes
        weight_bytes = trainable * p_bytes
        grad_bytes = trainable * p_bytes
        total = weight_bytes + grad_bytes + opt_bytes
        return {"weights": weight_bytes, "gradients": grad_bytes, "optimizer": opt_bytes, "total": total}
    else:
        trainable = params["lora_trainable"]
        frozen = params["lora_frozen"]
        opt_bytes = 2 * trainable * 4
        weight_bytes = frozen * p_bytes + trainable * p_bytes
        grad_bytes = trainable * p_bytes
        total = weight_bytes + grad_bytes + opt_bytes
        return {"weights": weight_bytes, "gradients": grad_bytes, "optimizer": opt_bytes, "total": total}


def compare_memory(cfg):
    full_res = compute_optimizer_bytes(cfg, "full")
    lora_res = compute_optimizer_bytes(cfg, "lora")
    ratio = lora_res["total"] / full_res["total"]
    diff = full_res["total"] - lora_res["total"]
    return {
        "full_total": full_res["total"],
        "lora_total": lora_res["total"],
        "ratio": ratio,
        "absolute_savings": diff,
    }
