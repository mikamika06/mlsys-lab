class RuntimeLimiter:
    def __init__(self, max_memory):
        self.max_memory = max_memory

    def check_and_apply(self, usage):
        if usage > self.max_memory:
            return "degrade"
        return "allow"

def degrade_gracefully(config, memory_limit):
    new_config = dict(config)
    if "context_length" in new_config:
        new_config["context_length"] = max(128, new_config["context_length"] // 2)
    return new_config

def run_safe_mode(configs):
    results = []
    for cfg in configs:
        limiter = RuntimeLimiter(max_memory=2000)
        est = cfg.get("context_length", 512) * 2
        if limiter.check_and_apply(est) == "degrade":
            cfg = degrade_gracefully(cfg, 2000)
        results.append(cfg)
    return results
