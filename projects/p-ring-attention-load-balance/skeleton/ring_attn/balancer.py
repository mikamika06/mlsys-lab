class RingBalancer:
    def __init__(self, world_size, seq_len):
        raise NotImplementedError

    def get_workload_per_step(self):
        raise NotImplementedError

    def rebalance(self, workloads):
        raise NotImplementedError

    def verify_equivalence(self, orig_output, new_output):
        raise NotImplementedError

    def measure_utilization(self, workloads):
        raise NotImplementedError

    def check_imbalance_threshold(self, workloads, threshold=0.6):
        raise NotImplementedError

    def predict_scaling(self, num_devices):
        raise NotImplementedError
