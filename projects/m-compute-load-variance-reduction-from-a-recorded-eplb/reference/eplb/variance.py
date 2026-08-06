import numpy as np


def compute_variance_reduction(expert_loads, initial_layout, final_layout, num_ranks):
    loads = np.asarray(expert_loads, dtype=np.float64)
    num_experts = len(loads)

    def calculate_rank_loads(layout):
        r_loads = np.zeros(num_ranks, dtype=np.float64)
        for e_id, ranks in enumerate(layout):
            e_load = loads[e_id] / len(ranks)
            for r in ranks:
                r_loads[r] += e_load
        return r_loads

    init_r_loads = calculate_rank_loads(initial_layout)
    final_r_loads = calculate_rank_loads(final_layout)

    var_initial = float(np.var(init_r_loads))
    var_final = float(np.var(final_r_loads))

    if var_initial > 0:
        var_reduction_ratio = float((var_initial - var_final) / var_initial)
    else:
        var_reduction_ratio = 0.0

    return {
        "var_initial": var_initial,
        "var_final": var_final,
        "var_reduction_ratio": var_reduction_ratio,
        "max_load_initial": float(np.max(init_r_loads)),
        "max_load_final": float(np.max(final_r_loads)),
        "initial_rank_loads": init_r_loads.tolist(),
        "final_rank_loads": final_r_loads.tolist()
    }
