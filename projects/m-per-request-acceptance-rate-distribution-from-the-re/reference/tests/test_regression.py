import sys
import numpy as np

sys.path.insert(0, ".")
from eagle_diag.metrics import compute_request_acceptance, compute_distribution_summary

LOG_SAMPLE = [
    {
        "request_id": "req-001",
        "draft_accepted_counts": [2, 3, 1, 4],
        "draft_proposed_counts": [4, 4, 4, 4],
    },
    {
        "request_id": "req-002",
        "draft_accepted_counts": [0, 1, 0, 2],
        "draft_proposed_counts": [3, 3, 3, 3],
    },
]


def test_acceptance_rate_bounds():
    stats = compute_request_acceptance(LOG_SAMPLE)
    for item in stats:
        rate = item["mean_acceptance_rate"]
        assert 0.0 <= rate <= 1.0, f"Acceptance rate out of bounds: {rate}"
        assert item["total_accepted"] <= item["total_proposed"]


def test_monotonic_accepted_tokens():
    for rec in LOG_SAMPLE:
        accepted = rec["draft_accepted_counts"]
        proposed = rec["draft_proposed_counts"]
        for a, p in zip(accepted, proposed):
            assert 0 <= a <= p, f"Accepted tokens {a} exceeds proposed {p}"
