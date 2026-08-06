import sys
sys.path.insert(0, ".")
from bench.metrics import robust_central_tendency
from bench.warmup import quantify_warmup_inflation
from bench.sync import validate_timing_agreement
import numpy as np

def test_median_robustness_to_outliers():
    clean = np.ones(100) * 50.0
    corrupted = clean.copy()
    corrupted[0] = 5000.0
    assert abs(robust_central_tendency(corrupted) - 50.0) < 1e-5

def test_warmup_inflation_sign():
    samples = np.concatenate([np.linspace(200.0, 100.0, 50), np.ones(100) * 100.0])
    inflation = quantify_warmup_inflation(samples, 50)
    assert inflation > 0.0

def test_sync_agreement_boundary():
    ev = [100.0, 200.0]
    wl = [105.0, 195.0]
    assert validate_timing_agreement(ev, wl, tolerance=0.1) is True
    wl_bad = [120.0, 180.0]
    assert validate_timing_agreement(ev, wl_bad, tolerance=0.1) is False
