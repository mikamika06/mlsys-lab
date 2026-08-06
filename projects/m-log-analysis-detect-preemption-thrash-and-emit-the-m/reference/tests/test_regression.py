import sys

sys.path.insert(0, ".")
from vllm_policy.scheduler import simulate


def test_aging_prevents_starvation():
    reqs = [
        {"id": 4, "arrival": 0, "prio": 0, "work": 2},
        {"id": 1, "arrival": 1, "prio": 10, "work": 2},
        {"id": 2, "arrival": 3, "prio": 10, "work": 2},
        {"id": 3, "arrival": 5, "prio": 10, "work": 2}
    ]

    comp_no_age = simulate(reqs, 0.0)
    comp_age = simulate(reqs, 5.0)

    assert comp_age[4] < comp_no_age[4], "Aging did not improve low priority completion time"
