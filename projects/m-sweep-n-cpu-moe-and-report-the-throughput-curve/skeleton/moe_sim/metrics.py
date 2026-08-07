def active_fraction(cfg):
    raise NotImplementedError

def vram_cost(cfg, ngl, n_cpu_experts):
    raise NotImplementedError

def latency(cfg, ngl, n_cpu_experts):
    raise NotImplementedError

def sweep_configs(cfg, max_vram):
    raise NotImplementedError
