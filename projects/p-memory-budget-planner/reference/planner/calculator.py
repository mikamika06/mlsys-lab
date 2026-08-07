class MemoryPlanner:
    def __init__(self, config):
        self.config = config
        self.p = config.get("num_params", 7000000000)
        self.h = config.get("hidden_size", 4096)
        self.l = config.get("num_layers", 32)
        self.s = config.get("seq_len", 2048)
        self.mb = config.get("micro_batch_size", 1)
        self.bpp = config.get("bytes_per_param", 2)
        self.ws = config.get("world_size", 1)
        self.zero = config.get("zero_stage", 0)
        self.ckpt = config.get("activation_checkpointing", False)
        self.offload = config.get("cpu_offload", False)

    def weights_memory(self):
        w = self.p * self.bpp
        if self.zero == 3:
            w = w / self.ws
        return int(w)

    def grads_memory(self):
        g = self.p * self.bpp
        if self.zero >= 2:
            g = g / self.ws
        return int(g)

    def opt_states_memory(self):
        if self.offload:
            return 0
        o = self.p * 12
        if self.zero >= 1:
            o = o / self.ws
        return int(o)

    def activations_memory(self):
        act = 34 * self.mb * self.s * self.h
        if not self.ckpt:
            act = act * self.l
        return int(act)

    def total_memory(self):
        return self.weights_memory() + self.grads_memory() + self.opt_states_memory() + self.activations_memory()

    def advise(self, limit):
        if self.total_memory() <= limit:
            return []
        suggestions = []
        if not self.ckpt:
            suggestions.append("activation_checkpointing")
        if self.zero < 3:
            suggestions.append("zero_stage")
        if self.mb > 1:
            suggestions.append("micro_batch_size")
        if not self.offload:
            suggestions.append("cpu_offload")
        return suggestions
