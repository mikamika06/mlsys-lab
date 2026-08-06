DTYPE_SIZES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 1,
}


def compute_adapter_bytes(peft_config: dict, base_model_shapes: dict) -> int:
    r = peft_config["r"]
    target_modules = peft_config["target_modules"]
    dtype_str = str(peft_config.get("torch_dtype", "float16")).lower()
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str.split(".")[-1]
    bytes_per_elem = DTYPE_SIZES.get(dtype_str, 2)

    total_params = 0
    for mod in target_modules:
        if mod in base_model_shapes:
            in_dim, out_dim = base_model_shapes[mod]
            total_params += (in_dim * r) + (r * out_dim)

    num_layers = base_model_shapes.get("num_layers", 1)
    return total_params * num_layers * bytes_per_elem
