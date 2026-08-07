def validate_batch_config(config: dict) -> bool:
    """Validate consistency of DeepSpeed batch size parameters."""
    raise NotImplementedError


def resolve_batch_config(config: dict) -> dict:
    """Deduce missing batch configuration parameter or validate complete dictionary."""
    raise NotImplementedError
