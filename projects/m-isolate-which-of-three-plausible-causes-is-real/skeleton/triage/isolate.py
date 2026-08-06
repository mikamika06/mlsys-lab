"""Isolation module for GGUF failure analysis."""


def isolate_root_cause(sample):
    """Analyze a single metric diagnostic dict and return the failure cause string.

    Returns one of: 'tokenizer_damage', 'quantization_damage', 'engine_failure'.
    """
    raise NotImplementedError
