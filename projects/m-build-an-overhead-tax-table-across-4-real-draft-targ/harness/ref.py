PAIRS_DATA = [
    {
        "pair_id": "d300m-t7b",
        "draft_step_ms": 1.2,
        "target_step_ms": 18.0,
        "verify_step_ms": {1: 18.2, 2: 18.5, 3: 18.9, 4: 19.4, 5: 20.0, 6: 20.7},
        "acceptance_probs": [0.82, 0.78, 0.71, 0.65, 0.58, 0.50],
    },
    {
        "pair_id": "d1b-t7b",
        "draft_step_ms": 3.1,
        "target_step_ms": 18.0,
        "verify_step_ms": {1: 18.2, 2: 18.5, 3: 18.9, 4: 19.4, 5: 20.0, 6: 20.7},
        "acceptance_probs": [0.91, 0.88, 0.84, 0.80, 0.75, 0.70],
    },
    {
        "pair_id": "d1.5b-t14b",
        "draft_step_ms": 4.5,
        "target_step_ms": 35.0,
        "verify_step_ms": {1: 35.4, 2: 35.9, 3: 36.5, 4: 37.2, 5: 38.0, 6: 39.0},
        "acceptance_probs": [0.88, 0.82, 0.76, 0.70, 0.62, 0.55],
    },
    {
        "pair_id": "d2b-t70b",
        "draft_step_ms": 6.0,
        "target_step_ms": 110.0,
        "verify_step_ms": {1: 110.8, 2: 111.7, 3: 112.8, 4: 114.0, 5: 115.5, 6: 117.2},
        "acceptance_probs": [0.95, 0.92, 0.89, 0.85, 0.81, 0.76],
    },
]


def calculate_expected_acceptance(acceptance_probs: list[float], gamma: int) -> dict:
    """Reference implementation of calculate_expected_acceptance."""
    probs = acceptance_probs[:gamma]
    cum_probs = []
    running = 1.0
    for p in probs:
        running *= p
        cum_probs.append(running)
    expected_accepted = 1.0 + sum(cum_probs)
    return {
        "gamma": gamma,
        "cum_probs": cum_probs,
        "expected_accepted": expected_accepted,
    }


def compute_pair_tax(pair_data: dict, gamma: int) -> dict:
    """Reference implementation of compute_pair_tax."""
    acc_info = calculate_expected_acceptance(pair_data["acceptance_probs"], gamma)
    expected_accepted = acc_info["expected_accepted"]

    draft_step_ms = pair_data["draft_step_ms"]
    target_step_ms = pair_data["target_step_ms"]
    verify_step_ms = pair_data["verify_step_ms"][gamma]

    total_draft_ms = draft_step_ms * gamma
    total_step_ms = total_draft_ms + verify_step_ms

    effective_latency_ms = total_step_ms / expected_accepted
    speedup = target_step_ms / effective_latency_ms
    overhead_tax = (effective_latency_ms - target_step_ms) / target_step_ms

    return {
        "pair_id": pair_data["pair_id"],
        "gamma": gamma,
        "total_step_ms": total_step_ms,
        "expected_accepted": expected_accepted,
        "effective_latency_ms": effective_latency_ms,
        "speedup": speedup,
        "overhead_tax": overhead_tax,
    }


def build_overhead_tax_table(pairs_data: list[dict], gammas: list[int]) -> list[dict]:
    """Reference implementation of build_overhead_tax_table."""
    table = []
    for pair in pairs_data:
        for g in gammas:
            row = compute_pair_tax(pair, g)
            table.append(row)
    return table


def find_optimal_gamma(pair_data: dict, max_gamma: int) -> dict:
    """Reference implementation of find_optimal_gamma."""
    best_tax = float("inf")
    best_res = None
    for g in range(1, max_gamma + 1):
        res = compute_pair_tax(pair_data, g)
        if res["overhead_tax"] < best_tax:
            best_tax = res["overhead_tax"]
            best_res = res
    return best_res
