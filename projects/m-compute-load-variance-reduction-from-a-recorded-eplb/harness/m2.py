import numpy as np
import ref
from eplb.redundant import rebalance_greedy_redundant


def check(workdir):
    cases = ref.generate_test_cases()
    max_err = 0.0

    for case in cases:
        loads = case["expert_loads"]
        num_ranks = case["num_ranks"]
        extra = 3

        got = rebalance_greedy_redundant(loads, num_ranks, extra)

        layout = got["layout"]
        calc_loads = np.zeros(num_ranks)
        for e, ranks in enumerate(layout):
            for r in ranks:
                calc_loads[r] += loads[e] / len(ranks)

        expected_max = float(np.max(calc_loads))
        err = abs(got["max_rank_load"] - expected_max)
        if err > max_err:
            max_err = err

    return {"rel_err": float(max_err)}
