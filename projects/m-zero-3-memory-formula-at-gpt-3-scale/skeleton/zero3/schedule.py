def build_schedule(num_layers: int, prefetch: int) -> list[tuple[str, int]]:
    """
    Yields operations in order: ('all_gather_fw', i), ('compute_fw', i), ('free_fw', i)
    and symmetrically for backward with ('all_gather_bw', i), ('compute_bw', i), ('reduce_scatter', i), ('free_bw', i).
    """
    raise NotImplementedError

def simulate_peak_memory(layers: list[int], schedule: list[tuple[str, int]]) -> int:
    """
    Returns the peak active bytes of fully materialized fp16 layers during the schedule.
    """
    raise NotImplementedError
