class ZeroEstimator:
    """Estimates memory and communication overhead for ZeRO configurations."""

    def __init__(self, num_params: int, bytes_per_param: int = 2, bytes_per_optim_state: int = 12):
        raise NotImplementedError

    def memory_zero1(self, world_size: int, act_mem_per_gpu: float) -> float:
        raise NotImplementedError

    def memory_zero2(self, world_size: int, act_mem_per_gpu: float) -> float:
        raise NotImplementedError

    def memory_zero3(self, world_size: int, act_mem_per_gpu: float) -> float:
        raise NotImplementedError

    def comm_bytes_per_step(self, stage: int, world_size: int) -> float:
        raise NotImplementedError

    def step_latency_with_offload(self, base_compute_time: float, stage: int, world_size: int, pcie_bandwidth_gbps: float, cpu_offload: bool) -> float:
        raise NotImplementedError
