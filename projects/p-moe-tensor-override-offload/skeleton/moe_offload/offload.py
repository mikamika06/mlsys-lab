class MoEOffloader:
    def __init__(self, tensor_sizes, memory_limit):
        raise NotImplementedError

    def measure_frequencies(self, traces):
        raise NotImplementedError

    def compute_rules(self, frequencies, memory_budget):
        raise NotImplementedError

    def evaluate_latency(self, offloaded, base_latency, penalty_factor=2.0):
        raise NotImplementedError

    def verify_output(self, ref_out, cand_out, tol=1e-5):
        raise NotImplementedError

    def check_constraints(self, offloaded, memory_budget, latency, max_latency):
        raise NotImplementedError
