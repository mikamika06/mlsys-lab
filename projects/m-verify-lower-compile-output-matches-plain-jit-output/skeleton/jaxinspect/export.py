import numpy as np


def verify_serialized_numerics(
    original_outputs: list[dict],
    deserialized_outputs: list[dict],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict:
    """Verify numeric match between original outputs and deserialized export outputs."""
    raise NotImplementedError
