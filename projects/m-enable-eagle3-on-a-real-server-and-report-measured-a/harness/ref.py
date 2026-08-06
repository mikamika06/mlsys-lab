import numpy as np

CONFIGS = [
    {
        "model": "llama-8b",
        "top_k": 4,
        "depth": 3,
        "num_speculative_tokens": 5
    },
    {
        "model": "llama-70b",
        "top_k": 8,
        "depth": 4,
        "num_speculative_tokens": 7
    },
    {
        "model": "mistral-7b",
        "top_k": 3,
        "depth": 3,
        "num_speculative_tokens": 4
    }
]

def build_config(cfg):
    return {
        "model_name": cfg["model"],
        "speculative_model": f"{cfg['model']}-eagle3",
        "speculative_draft_tensor_parallel_size": 1,
        "num_speculative_tokens": cfg["num_speculative_tokens"],
        "speculative_max_model_len": 4096,
        "tree_config": {
            "top_k": cfg["top_k"],
            "depth": cfg["depth"]
        }
    }

RUNS = [
    {
        "accepted": 120,
        "total": 200,
        "baseline_tpot": 35.0,
        "eagle_tpot": 22.0
    },
    {
        "accepted": 300,
        "total": 450,
        "baseline_tpot": 40.0,
        "eagle_tpot": 25.0
    },
    {
        "accepted": 80,
        "total": 150,
        "baseline_tpot": 30.0,
        "eagle_tpot": 20.0
    }
]

def simulate_server(run_cfg):
    acc = run_cfg["accepted"]
    tot = run_cfg["total"]
    return {
        "accepted_tokens": acc,
        "total_tokens": tot,
        "acceptance_rate": acc / tot
    }

def compute_metrics(run_cfg, sim_out):
    ar = sim_out["acceptance_rate"]
    lat_ratio = run_cfg["eagle_tpot"] / run_cfg["baseline_tpot"]
    tpot_gain = run_cfg["baseline_tpot"] / run_cfg["eagle_tpot"]
    return {
        "acceptance_rate": ar,
        "latency_ratio": lat_ratio,
        "tpot_gain": tpot_gain
    }
