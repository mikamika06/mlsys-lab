import numpy as np
import sys
sys.path.insert(0, ".")
from ropespectrum.spectrum import reconstruct_spectrum, classify_dims
from ropespectrum.overflow import simulate_overflow

def test_spectrum_shape_and_monotonicity():
    freqs = reconstruct_spectrum(128, 10000.0)
    assert len(freqs) == 64
    assert np.all(freqs[:-1] >= freqs[1:])

def test_classification_coverage():
    classes = classify_dims(64, 10000.0, 0.01)
    assert len(classes) == 32
    assert set(classes).issubset({"high", "low"})

def test_overflow_detection():
    pos = np.array([100, 5000])
    _, overflow = simulate_overflow(pos, 64, 10000.0, 4096)
    assert not overflow[0]
    assert overflow[1]
