import numpy as np

def get_workload():
    np.random.seed(42)
    return np.random.randn(64, 64).astype(np.float32)

def oracle_thread_scaling(workload, thread_counts):
    results = {}
    for t in thread_counts:
        lat = 210.0 / (t ** 0.5) + max(0.0, (t - 4) * 5.0)
        results[t] = float(lat)
    return results

def oracle_optimal_threads(thread_counts):
    w = get_workload()
    res = oracle_thread_scaling(w, thread_counts)
    return min(res, key=res.get)

def oracle_arena_config(strategy):
    return {"enable_arena": True, "strategy": strategy, "chunk_size": 1024 * 1024}

def oracle_io_binding(tensor):
    return {"bound": True, "zero_copy": True, "shape": tensor.shape}

def oracle_opt_level(complexity):
    return 99 if complexity > 50 else 1

def oracle_run_latency(config):
    threads = config.get("intra_threads", 1)
    arena = config.get("enable_arena", False)
    io_bind = config.get("io_binding", False)
    opt = config.get("opt_level", 0)
    lat = 250.0
    lat -= threads * 15.0 if threads <= 4 else 60.0 - (threads - 4) * 5.0
    if arena:
        lat -= 30.0
    if io_bind:
        lat -= 40.0
    if opt >= 99:
        lat -= 50.0
    elif opt > 0:
        lat -= 20.0
    return max(45.0, float(lat))
