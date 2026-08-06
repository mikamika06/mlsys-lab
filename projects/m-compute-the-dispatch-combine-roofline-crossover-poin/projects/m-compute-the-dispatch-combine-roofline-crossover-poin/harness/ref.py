import random


def get_test_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(5):
        hidden_dim = rng.choice([2048, 4096, 8192])
        num_experts = rng.choice([8, 16, 32])
        comm_bw = rng.choice([100.0, 200.0, 400.0])
        tflops = rng.choice([150.0, 300.0, 600.0])
        cases.append({
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "comm_bw": comm_bw,
            "tflops": tflops
        })
    return cases


def get_packing_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(5):
        num_experts = rng.choice([16, 32])
        num_gpus = rng.choice([4, 8])
        loads = [rng.randint(100, 1000) for _ in range(num_experts)]
        cases.append({
            "loads": loads,
            "num_gpus": num_gpus
        })
    return cases


def compute_crossover(hidden_dim, num_experts, comm_bw, tflops):
    bytes_per_token = hidden_dim * 2
    comm_cost_per_token = bytes_per_token / (comm_bw * 1e9)
    compute_ops_per_token = 2.0 * hidden_dim * 2.0
    compute_cost_per_token = compute_ops_per_token / (tflops * 1e12)
    crossover_tokens = int(compute_cost_per_token / comm_cost_per_token * num_experts)
    return max(1, crossover_tokens)


def pack_experts(loads, num_gpus):
    indexed = sorted(enumerate(loads), key=lambda x: x[1], reverse=True)
    gpus = [[] for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus
    for idx, load in indexed:
        target = min(range(num_gpus), key=lambda g: gpu_loads[g])
        gpus[target].append(idx)
        gpu_loads[target] += load
    return gpus
