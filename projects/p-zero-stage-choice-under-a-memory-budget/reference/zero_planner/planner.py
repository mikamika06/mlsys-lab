from zero_planner.estimator import ZeroEstimator


class ZeroPlanner:
    """Selects the best ZeRO configuration given budget and hardware constraints."""

    def __init__(self, num_params: int, bytes_per_param: int = 2, bytes_per_optim_state: int = 12):
        self.estimator = ZeroEstimator(num_params, bytes_per_param, bytes_per_optim_state)

    def select_config(self, budget_gb: float, world_size: int, act_mem_per_batch_item: float, max_batch_size: int, pcie_bandwidth_gbps: float):
        budget_bytes = budget_gb * (1024 ** 3)
        best_config = None
        min_latency = float("inf")

        for bs in range(max_batch_size, 0, -1):
            act_mem = bs * act_mem_per_batch_item
            for stage in [1, 2, 3]:
                for offload in [False, True]:
                    if stage == 1:
                        mem = self.estimator.memory_zero1(world_size, act_mem)
                    elif stage == 2:
                        mem = self.estimator.memory_zero2(world_size, act_mem)
                    else:
                        mem = self.estimator.memory_zero3(world_size, act_mem)

                    if offload:
                        if stage in (1, 2):
                            off_bytes = (self.estimator.num_params * self.estimator.bytes_per_optim_state) / world_size
                        else:
                            off_bytes = ((self.estimator.num_params * self.estimator.bytes_per_optim_state) + (self.estimator.num_params * self.estimator.bytes_per_param)) / world_size
                        mem -= off_bytes

                    if mem <= budget_bytes:
                        base_comp = 0.1 * bs
                        lat = self.estimator.step_latency_with_offload(base_comp, stage, world_size, pcie_bandwidth_gbps, offload)
                        if lat < min_latency:
                            min_latency = lat
                            best_config = {
                                "stage": stage,
                                "batch_size": bs,
                                "cpu_offload": offload,
                                "predicted_mem_bytes": mem,
                                "predicted_latency": lat
                            }
            if best_config is not None:
                break

        return best_config

    def verify_trace(self, trace_data: dict, predicted_mem_bytes: float) -> bool:
        actual_peak = trace_data.get("peak_memory_bytes", 0)
        if actual_peak <= 0:
            return False
        diff = abs(actual_peak - predicted_mem_bytes) / actual_peak
        return diff <= 0.15

    def predict_doubled_gpus(self, current_world_size: int, stage: int, act_mem_per_gpu: float) -> dict:
        new_world_size = current_world_size * 2
        if stage == 1:
            mem_fn = self.estimator.memory_zero1
        elif stage == 2:
            mem_fn = self.estimator.memory_zero2
        else:
            mem_fn = self.estimator.memory_zero3

        old_mem = mem_fn(current_world_size, act_mem_per_gpu)
        new_mem = mem_fn(new_world_size, act_mem_per_gpu)

        old_comm = self.estimator.comm_bytes_per_step(stage, current_world_size)
        new_comm = self.estimator.comm_bytes_per_step(stage, new_world_size)

        return {
            "new_world_size": new_world_size,
            "memory_bytes": new_mem,
            "memory_saved_bytes": old_mem - new_mem,
            "comm_bytes": new_comm
        }
