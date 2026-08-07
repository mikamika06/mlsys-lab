import numpy as np
from snrchk.model import BundledProgram, run_eager
from snrchk.analyzer import compute_layer_snrs, bisect_divergence

def test_snr_perfect_match():
    np.random.seed(0)
    x = np.random.randn(4, 16)
    weights = [np.random.randn(16, 16) for _ in range(5)]
    eager = run_eager(weights, x)
    prog = BundledProgram(weights, diverge_layer=-1)
    exported = prog.run_exported(x)
    snrs = compute_layer_snrs(eager, exported)
    assert all(s > 50.0 for s in snrs)

def test_bisect_divergence_found():
    snrs = [30.0, 28.0, 15.0, 10.0]
    assert bisect_divergence(snrs, threshold=20.0) == 2

def test_bisect_divergence_none():
    snrs = [30.0, 35.0, 40.0]
    assert bisect_divergence(snrs, threshold=20.0) == -1
