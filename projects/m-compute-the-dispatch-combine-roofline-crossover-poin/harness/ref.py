CONFIGS = [
    {"num_ranks": 8, "hidden_dim": 4096, "ffn_inter_dim": 14336, "bus_gbps": 900.0, "compute_tflops": 989.0, "tokens": 512},
    {"num_ranks": 16, "hidden_dim": 8192, "ffn_inter_dim": 28672, "bus_gbps": 450.0, "compute_tflops": 1979.0, "tokens": 16384},
    {"num_ranks": 4, "hidden_dim": 2048, "ffn_inter_dim": 8192, "bus_gbps": 300.0, "compute_tflops": 312.0, "tokens": 128},
    {"num_ranks": 32, "hidden_dim": 7168, "ffn_inter_dim": 20480, "bus_gbps": 1800.0, "compute_tflops": 1000.0, "tokens": 8192},
]

PACK_CONFIGS = [
    {"loads": [100, 90, 80, 70, 60, 50, 40, 30], "ranks": 4, "exp_mem": 500, "budget": 1500},
    {"loads": [500, 10, 10, 10, 500, 10, 10, 10], "ranks": 2, "exp_mem": 200, "budget": 1000},
    {"loads": [120, 110, 100, 90, 80, 70], "ranks": 3, "exp_mem": 300, "budget": 900},
    {"loads": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "ranks": 5, "exp_mem": 100, "budget": 300},
]


def ref_crossover(cfg):
    num_ranks = cfg["num_ranks"]
    h = cfg["hidden_dim"]
    f = cfg["ffn_inter_dim"]
    bw = cfg["bus_gbps"]
    tflops = cfg["compute_tflops"]
    comm_bytes = 2 * ((num_ranks - 1) / num_ranks) * 2 * h * 2
    t_comm = comm_bytes / (bw * 1e9)
    flops = 4 * h * f
    t_comp = flops / (tflops * 1e12)
    return t_comm / t_comp


def ref_classify(cfg):
    crossover = ref_crossover(cfg)
    return "communication_bound" if cfg["tokens"] < crossover else "compute_bound"


def ref_pack(cfg):
    loads = cfg["loads"]
    ranks = cfg["ranks"]
    exp_mem = cfg["exp_mem"]
    budget = cfg["budget"]
    max_cap = budget // exp_mem
    indexed = sorted(enumerate(loads), key=lambda x: x[1], reverse=True)
    r_loads = [0] * ranks
    r_counts = [0] * ranks
    res = {}
    for exp_id, load in indexed:
        feasible = [r for r in range(ranks) if r_counts[r] < max_cap]
        best = min(feasible, key=lambda r: (r_loads[r], r_counts[r]))
        res[exp_id] = best
        r_loads[best] += load
        r_counts[best] += 1
    return [res[i] for i in range(len(loads))]
