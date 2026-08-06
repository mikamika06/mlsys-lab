import re


def count_trainable_parameters(model_layout, target_modules, lora_r):
    total_params = 0
    trainable_params = 0

    for module in model_layout:
        name = module["name"]
        in_dim = module["in_features"]
        out_dim = module["out_features"]

        base_count = in_dim * out_dim
        total_params += base_count

        matched = False
        for target in target_modules:
            if target in name or re.search(r"\b" + re.escape(target) + r"\b", name):
                matched = True
                break

        if matched:
            adapter_count = lora_r * in_dim + lora_r * out_dim
            trainable_params += adapter_count
            total_params += adapter_count

    return {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": float(trainable_params / total_params) if total_params > 0 else 0.0,
    }
