from preempt.model import recompute_cost, swap_cost

class PreemptionPolicy:
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.get("hidden_size", 4096)
        self.num_layers = config.get("num_layers", 32)
        self.tflops = config.get("tflops", 300.0)
        self.bytes_per_token = config.get("bytes_per_token", 131072)
        self.pcie_bw = config.get("pcie_bandwidth_gbps", 32.0)
        self.mode = config.get("mode", "adaptive")

    def decide(self, request, cluster_state):
        if self.mode == "recompute":
            return "recompute"
        if self.mode == "swap":
            return "swap"

        ctx_len = request.get("context_len", 1024)
        rc = recompute_cost(ctx_len, self.hidden_size, self.num_layers, self.tflops)
        sc = swap_cost(ctx_len, self.bytes_per_token, self.pcie_bw)
        return "swap" if sc < rc else "recompute"
