from ftmem.model import count_base_params, get_layer_shapes


def count_lora_params(config: dict, lora_config: dict) -> int:
    r = lora_config["r"]
    targets = lora_config["target_modules"]
    shapes = get_layer_shapes(config)
    layers = config["num_hidden_layers"]
    per_layer = sum(
        r * (din + dout)
        for m in targets
        if m in shapes
        for din, dout in [shapes[m]]
    )
    return layers * per_layer


def count_trainable_params(config: dict, lora_config: dict | None = None) -> int:
    if lora_config is None:
        return count_base_params(config)
    return count_lora_params(config, lora_config)
