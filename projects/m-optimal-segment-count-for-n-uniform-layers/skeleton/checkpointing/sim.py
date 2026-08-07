def baseline(n_layers: int, layer_mem: int, fwd_time: int, bwd_time: int) -> dict:
    raise NotImplementedError


def simulate_checkpointing(n_layers: int, segments: int, layer_mem: int, fwd_time: int, bwd_time: int) -> dict:
    raise NotImplementedError


def optimal_segments(n_layers: int) -> int:
    raise NotImplementedError
