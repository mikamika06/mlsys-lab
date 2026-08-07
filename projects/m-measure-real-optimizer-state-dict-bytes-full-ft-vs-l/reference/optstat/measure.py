DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
}


def measure_optimizer_bytes(state_dict):
    total = 0
    state = state_dict.get("state", {})
    for p_id, p_state in state.items():
        if not isinstance(p_state, dict):
            continue
        for k, v in p_state.items():
            if isinstance(v, dict) and "shape" in v and "dtype" in v:
                elems = 1
                for d in v["shape"]:
                    elems *= d
                total += elems * DTYPE_BYTES.get(str(v["dtype"]), 4)
            elif isinstance(v, (int, float)):
                total += 8
    return total


def profile_model_optimizer_bytes(model_config, mode):
    trainable_params = 0
    total_params = 0

    for layer in model_config["layers"]:
        in_dim = layer["in_features"]
        out_dim = layer["out_features"]
        base_count = in_dim * out_dim
        total_params += base_count

        if mode == "full":
            trainable_params += base_count
        elif mode == "lora":
            rank = layer.get("lora_rank", 8)
            lora_count = (in_dim * rank) + (rank * out_dim)
            trainable_params += lora_count
            if not layer.get("frozen", True):
                total_params += lora_count

    opt_bytes = trainable_params * 8
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "optimizer_bytes": opt_bytes,
    }
