import numpy as np


def apply_per_row_lora(x, adapter_ids, lora_a, lora_b, scaling):
    """Apply adapter matrices per-row based on assigned adapter IDs."""
    raise NotImplementedError


def verify_batched_lora(x, adapter_ids, lora_a, lora_b, scaling, expected_out, atol=1e-5):
    """Verify that batched per-row adapter output matches expected tensor."""
    raise NotImplementedError
