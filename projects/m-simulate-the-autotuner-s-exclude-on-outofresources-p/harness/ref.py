class OutOfResources(Exception):
    pass


def is_dominated(config, oom_configs, resource_keys):
    for oom in oom_configs:
        if all(config[k] >= oom[k] for k in resource_keys):
            return True
    return False


def autotune(configs, evaluate, resource_keys):
    oom_configs = []
    best_idx = -1
    best_time = float('inf')

    for i, config in enumerate(configs):
        if is_dominated(config, oom_configs, resource_keys):
            continue
        try:
            time = evaluate(config)
            if time < best_time:
                best_time = time
                best_idx = i
        except OutOfResources:
            oom_configs.append(config)

    return best_idx


def generate_configs():
    configs = []
    for m in [32, 64, 128]:
        for n in [32, 64, 128]:
            for w in [4, 8]:
                configs.append({"BLOCK_M": m, "BLOCK_N": n, "num_warps": w})
    return configs


def make_evaluator(limit):
    metrics = {"evals": 0}
    def evaluate(config):
        metrics["evals"] += 1
        size = config["BLOCK_M"] * config["BLOCK_N"] * config["num_warps"]
        if size >= limit:
            raise OutOfResources("OOM")

        # Pseudo-time: larger configs run faster, but have a tiny penalty
        return 10000.0 / size + size * 0.001

    return evaluate, metrics
