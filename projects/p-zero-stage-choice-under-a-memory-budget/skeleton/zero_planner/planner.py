class ZeroPlanner:
    """Selects the best ZeRO configuration given budget and hardware constraints."""

    def __init__(self, num_params: int, bytes_per_param: int = 2, bytes_per_optim_state: int = 12):
        raise NotImplementedError

    def select_config(self, budget_gb: float, world_size: int, act_mem_per_batch_item: float, max_batch_size: int, pcie_bandwidth_gbps: float):
        raise NotImplementedError

    def verify_trace(self, trace_data: dict, predicted_mem_bytes: float) -> bool:
        raise NotImplementedError

    def predict_doubled_gpus(self, current_world_size: int, stage: int, act_mem_per_gpu: float) -> dict:
        raise NotImplementedError
