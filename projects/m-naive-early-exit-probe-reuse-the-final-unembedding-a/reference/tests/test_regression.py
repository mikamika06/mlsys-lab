import sys
import numpy as np

sys.path.insert(0, ".")
from earlyexit.probe import compute_early_exit_agreement, sweep_and_compare


def test_agreement_range():
    rng = np.random.default_rng(123)
    h = rng.standard_normal((8, 32))
    w = rng.standard_normal((64, 32))
    f = rng.standard_normal((8, 64))
    val = compute_early_exit_agreement(h, w, f)
    assert 0.0 <= val <= 1.0


def test_sweep_output_structure():
    rng = np.random.default_rng(123)
    d = {2: rng.standard_normal((4, 16)), 4: rng.standard_normal((4, 16))}
    w = rng.standard_normal((32, 16))
    f = rng.standard_normal((4, 32))
    tbl = {2: 0.4, 4: 0.7}
    res = sweep_and_compare(d, w, f, tbl)
    assert "agreements" in res
    assert "mean_diff" in res
    assert isinstance(res["mean_diff"], float)
    assert len(res["agreements"]) == 2
