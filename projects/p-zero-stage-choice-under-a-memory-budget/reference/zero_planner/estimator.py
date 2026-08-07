class ZeroEstimator:
    """Estimates memory and communication overhead for ZeRO configurations."""

    def __init__(self, num_params: int, bytes_per_param: int = 2, bytes_per_optim_state: int = 12):
        self.num_params = num_params
        self.bytes_per_param = bytes_per_param
        self.bytes_per_optim_state = bytes_per_optim_state

    def memory_zero1(self, world_size: int, act_mem_per_gpu: float) -> float:
        w_mem = self.num_params * self.bytes_per_param
        g_mem = self.num_params * self.bytes_per_param
        o_mem = (self.num_params * self.bytes_per_optim_state) / world_size
        return w_mem + g_mem + o_mem + act_mem_per_gpu

    def memory_zero2(self, world_size: int, act_mem_per_gpu: float) -> float:
        w_mem = self.num_params * self.bytes_per_param
        g_mem = (self.num_params * self.bytes_per_param) / world_size
        o_mem = (self.num_params * self.bytes_per_optim_state) / world_size
        return w_mem + g_mem + o_mem + act_mem_per_gpu

    def memory_zero3(self, world_size: int, act_mem_per_gpu: float) -> float:
        w_mem = (self.num_params * self.bytes_per_param) / world_size
        g_mem = (self.num_params * self.bytes_per_param) / world_size
        o_mem = (self.num_params * self.bytes_per_optim_state) / world_size
        return w_mem + g_mem + o_mem + act_mem_per_gpu

    def comm_bytes_per_step(self, stage: int, world_size: int) -> float:
        if world_size <= 1:
            return 0.0
        psi = self.num_params * self.bytes_per_param
        scale = (world_size - 1) / world_size
        if stage in (1, 2):
            return 2.0 * psi * scale
        elif stage == 3:
            return 3.0 * psi * scale
        raise ValueError(f"Unknown stage {stage}")

    def step_latency_with_offload(self, base_compute_time: float, stage: int, world_size: int, pcie_bandwidth_gbps: float, cpu_offload: bool) -> float:
        if not cpu_offload:
            return base_compute_time
        bw_bytes_sec = pcie_bandwidth_gbps * 1e9
        if stage in (1, 2):
            offloaded_bytes = (self.num_params * self.bytes_per_optim_state) / world_size
        elif stage == 3:
            offloaded_bytes = ((self.num_params * self.bytes_per_optim_state) + (self.num_params * self.bytes_per_param)) / world_size
        else:
            offloaded_bytes = 0.0
        transfer_time = (2.0 * offloaded_bytes) / bw_bytes_sec
        return base_compute_time + transfer_time
