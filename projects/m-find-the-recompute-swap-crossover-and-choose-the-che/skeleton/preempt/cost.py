def compute_recompute_time(tokens_to_recompute: int, compute_tflops: float, model_params_b: float) -> float:
    raise NotImplementedError


def compute_swap_time(num_blocks: int, block_size_bytes: int, pcie_bandwidth_gbps: float) -> float:
    raise NotImplementedError


def choose_preemption_mode(tokens_to_recompute: int, num_blocks: int, block_size_bytes: int, compute_tflops: float, model_params_b: float, pcie_bandwidth_gbps: float) -> str:
    raise NotImplementedError
