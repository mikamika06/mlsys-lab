import numpy as np

def generate_fixtures():
    np.random.seed(42)

    configs = [
        {
            "id": "matched_lightweight_draft",
            "draft_latency": 1.5,
            "target_verify_latency": 10.0,
            "target_step_latency": 10.0,
            "K": 5,
            "acceptance_rate": 0.85
        },
        {
            "id": "oversized_draft_heavy_overhead",
            "draft_latency": 6.0,
            "target_verify_latency": 12.0,
            "target_step_latency": 10.0,
            "K": 5,
            "acceptance_rate": 0.40
        },
        {
            "id": "low_acceptance_mismatch",
            "draft_latency": 2.0,
            "target_verify_latency": 10.0,
            "target_step_latency": 10.0,
            "K": 4,
            "acceptance_rate": 0.15
        },
        {
            "id": "optimal_speculative",
            "draft_latency": 0.8,
            "target_verify_latency": 10.0,
            "target_step_latency": 10.0,
            "K": 6,
            "acceptance_rate": 0.90
        }
    ]

    pld_cases = [
        {
            "prompt": [10, 20, 30, 40, 20, 30, 40],
            "max_k": 8,
            "n_gram_len": 3
        },
        {
            "prompt": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "max_k": 5,
            "n_gram_len": 2
        },
        {
            "prompt": [100, 200, 300, 200, 300, 200, 300],
            "max_k": 10,
            "n_gram_len": 2
        }
    ]

    in_domain_evals = [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([10, 20, 30, 40, 50], [10, 20, 30, 99, 50]),
        ([100, 200, 300, 400], [100, 200, 300, 400])
    ]

    ood_evals = [
        ([1, 2, 3, 4, 5], [1, 99, 3, 4, 5]),
        ([10, 20, 30, 40, 50], [99, 20, 30, 40, 50]),
        ([100, 200, 300, 400], [100, 99, 300, 400])
    ]

    return {
        "configs": configs,
        "pld_cases": pld_cases,
        "in_domain_evals": in_domain_evals,
        "ood_evals": ood_evals
    }
