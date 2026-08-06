KERNELS = [
    {
        "kernel_id": "k_reduce_v1",
        "warp_stats": {
            "stall_barrier": 4500,
            "stall_mio": 1200,
            "stall_not_selected": 800,
            "active": 300
        },
        "branches": [
            {"branch_id": "b_left", "divergence_score": 12, "execution_cycles": 200},
            {"branch_id": "b_right", "divergence_score": 85, "execution_cycles": 950}
        ]
    },
    {
        "kernel_id": "k_reduce_v2",
        "warp_stats": {
            "stall_barrier": 6200,
            "stall_mio": 900,
            "stall_not_selected": 400,
            "active": 150
        },
        "branches": [
            {"branch_id": "b_alpha", "divergence_score": 92, "execution_cycles": 1100},
            {"branch_id": "b_beta", "divergence_score": 5, "execution_cycles": 150}
        ]
    }
]

def identify_imbalanced_branch(kernel_data):
    branches = kernel_data["branches"]
    return max(branches, key=lambda b: b["divergence_score"])["branch_id"]

def verify_sync_removal(baseline_stats, modified_stats, speedup_ratio):
    base_barrier = baseline_stats["warp_stats"]["stall_barrier"]
    mod_barrier = modified_stats["warp_stats"]["stall_barrier"]
    drop_ratio = (base_barrier - mod_barrier) / base_barrier
    return drop_ratio >= 0.2 and speedup_ratio >= 1.15
