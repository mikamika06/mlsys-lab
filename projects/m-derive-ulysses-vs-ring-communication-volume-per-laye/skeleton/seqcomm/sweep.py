from typing import Any, Dict, List


def sweep_comm_costs(
    seq_lens: List[int],
    hidden_dims: List[int],
    world_sizes: List[int],
    head_counts: List[int],
    dtype_bytes: int = 2,
) -> List[Dict[str, Any]]:
    """Sweep communication volumes across Ulysses, Ring, and USP configurations."""
    raise NotImplementedError
