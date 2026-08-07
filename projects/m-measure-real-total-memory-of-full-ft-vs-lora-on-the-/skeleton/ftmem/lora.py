def count_lora_params(config: dict, lora_config: dict) -> int:
    raise NotImplementedError


def count_trainable_params(config: dict, lora_config: dict | None = None) -> int:
    raise NotImplementedError
