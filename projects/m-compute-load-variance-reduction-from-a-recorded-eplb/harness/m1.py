import numpy as np
import ref
from eplb.variance import compute_variance_reduction


def check(workdir):
    cases = ref.generate_test_cases()
    max_err = 0.0

    for case in cases:
        loads = case["expert_loads"]
        num_ranks = case["num_ranks"]
        init_l = case["initial_layout"]
        final_l = case["final_layout"]

        got = compute_variance_reduction(loads, init_l, final_l, num_ranks)

        r_loads_init = np.zeros(num_ranks)
        for e, ranks in enumerate(init_l):
            for r in ranks:
                r_loads_init[r] += loads[e] / len(ranks)
        var_init = float(np.var(r_loads_init))

        r_loads_final = np.zeros(num_ranks)
        for e, ranks in enumerate(final_l):
            for r in ranks:
                r_loads_final[r] += loads[e] / len(ranks)
        var_final = float(np.var(r_loads_final))

        expected_ratio = (var_init - var_final) / var_init if var_init > 0 else 0.0

        err = abs(got["var_reduction_ratio"] - expected_ratio)
        if err > max_err:
            max_err = err

    return {"rel_err": float(max_err)}
