import sys

sys.path.insert(0, ".")
from pipesched.metrics import compute_bubble_fraction
from pipesched.schedule import identify_schedule


def test_identify_gpipe():
    actions = [
        {"type": "FORWARD", "microbatch": 0},
        {"type": "FORWARD", "microbatch": 1},
        {"type": "BACKWARD", "microbatch": 0},
        {"type": "BACKWARD", "microbatch": 1},
    ]
    assert identify_schedule(actions) == "gpipe"


def test_identify_zero_bubble():
    actions = [
        {"type": "FORWARD", "microbatch": 0},
        {"type": "BACKWARD_WEIGHT", "microbatch": 0},
        {"type": "BACKWARD", "microbatch": 0},
    ]
    assert identify_schedule(actions) == "zero_bubble"


def test_bubble_fraction_computation():
    logs = [
        "RANK 0: compute_time=100.00ms",
        "RANK 0: idle_bubble_time=25.00ms",
    ]
    frac = compute_bubble_fraction(logs)
    assert abs(frac - 0.2) < 1e-5
