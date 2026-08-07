def estimate_activation_memory(
    config: dict,
    batch_size: int = 1,
    seq_len: int = 512,
    activation_checkpointing: bool = False,
) -> int:
    raise NotImplementedError


def estimate_memory_footprint(
    config: dict,
    mode: str,
    lora_config: dict | None = None,
    batch_size: int = 1,
    seq_len: int = 512,
    activation_checkpointing: bool = False,
) -> dict:
    raise NotImplementedError
