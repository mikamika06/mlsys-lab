import numpy as np
from bfacc.accumulation import Accumulator, compute_relative_error
from bfacc.repro import diagnose_loss_spike


def test_accumulation_precision():
    acc = Accumulator((100,), dtype=np.float32)
    rng = np.random.RandomState(42)
    for _ in range(50):
        delta = rng.randn(100).astype(np.float32) * 1e-3
        acc.update(delta)
    naive, comp = acc.get_values()
    err = compute_relative_error(naive, comp)
    assert err > 0.0, "Expected precision degradation in naive update"


def test_loss_spike_detection():
    history = [1.0, 0.95, 0.92, 0.90, 4.5, 2.1, 1.5]
    diag = diagnose_loss_spike(history, threshold=2.0)
    assert diag["spiked"] is True
    assert diag["step"] == 4
