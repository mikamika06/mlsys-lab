class RuntimeConfigurator:
    def __init__(self, hardware_info):
        self.hw = hardware_info
        self.threads = 1
        self.arena = True
        self.io_binding = False
        self.graph_opt = 1
        self.measured_latencies = {}

    def measure_threads(self, thread_counts):
        for t in thread_counts:
            simulated_latency = 200.0 / t + float(t) * 2.0
            self.measured_latencies[t] = simulated_latency
        best_t = min(self.measured_latencies, key=self.measured_latencies.get)
        self.threads = best_t
        return self.measured_latencies

    def configure_arena(self, enable=True):
        self.arena = enable
        return self.arena

    def use_io_binding(self, enable=True):
        self.io_binding = enable
        return self.io_binding

    def set_graph_optimization(self, level):
        self.graph_opt = level
        return self.graph_opt

    def get_latency(self):
        base = 120.0 / max(1, self.threads) + (10.0 if not self.arena else 0.0) + (15.0 if not self.io_binding else 0.0) - (self.graph_opt * 5.0)
        return max(40.0, base)

    def build_config(self):
        return {
            "intra_op_num_threads": self.threads,
            "arena_extend_strategy": "kNextPowerOfTwo" if self.arena else "kSameAsRequested",
            "graph_optimization_level": self.graph_opt,
            "use_io_binding": self.io_binding
        }
