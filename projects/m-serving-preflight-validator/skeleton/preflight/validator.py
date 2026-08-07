"""Preflight validation logic for vLLM deployment configurations."""


def estimate_memory_bytes(model_config, quant_config, num_gpus):
    """Estimate total memory required per GPU in bytes."""
    raise NotImplementedError


def validate_fit(model_config, quant_config, gpu_specs):
    """Validate if quantized model fits on given GPU configuration."""
    raise NotImplementedError
