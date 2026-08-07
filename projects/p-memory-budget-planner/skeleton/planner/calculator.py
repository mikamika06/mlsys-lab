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
        raise NotImplementedError

    def grads_memory(self):
        raise NotImplementedError

    def opt_states_memory(self):
        raise NotImplementedError

    def activations_memory(self):
        raise NotImplementedError

    def total_memory(self):
        raise NotImplementedError

    def advise(self, limit):
        raise NotImplementedError
