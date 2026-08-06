import sys
import numpy as np

sys.path.insert(0, ".")
from throttling.detect import detect_transition
from throttling.metrics import severity_score


def test_transition_within_bounds():
    rng = np.random.default_rng(123)
    trace = np.ones(500)
    trace[250:] *= 1.4
    idx = detect_transition(trace)
    assert 230 <= idx <= 270, f"transition detected at {idx}, expected around 250"


def test_severity_calculation():
    trace = np.array([10.0] * 50 + [20.0] * 50)
    sev = severity_score(trace, 50)
    assert np.isclose(sev, 2.0), f"severity score {sev} != 2.0"
