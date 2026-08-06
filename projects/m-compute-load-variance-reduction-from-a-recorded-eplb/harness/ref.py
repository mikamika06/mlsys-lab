import numpy as np


def generate_test_cases():
    np.random.seed(42)
    cases = []
    for i in range(5):
        num_experts = 8
        num_ranks = 4
        expert_loads = np.random.uniform(10.0, 100.0, size=num_experts).tolist()

        init_layout = [[i % num_ranks] for i in range(num_experts)]

        final_layout = [[] for _ in range(num_experts)]
        for e in range(num_experts):
            if e < 2:
                final_layout[e] = [e % num_ranks, (e + 1) % num_ranks]
            else:
                final_layout[e] = [e % num_ranks]

        cases.append({
            "expert_loads": expert_loads,
            "num_ranks": num_ranks,
            "initial_layout": init_layout,
            "final_layout": final_layout
        })
    return cases
