def context_multiplier(
    batch_size: int,
    hidden_size: int,
    element_bytes: int,
    gpu_budget_bytes: int,
    sequence_length: int,
) -> float:
    standard_bytes = (
        batch_size
        * sequence_length
        * hidden_size
        * element_bytes
    )
    offloaded_bytes = (
        batch_size
        * hidden_size
        * element_bytes
    )
    return float(standard_bytes / offloaded_bytes)
