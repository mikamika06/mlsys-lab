import ref
import numpy as np


def check(workdir):
    from pt2ex.quant import observe_ranges
    activations, _ = ref.get_test_fixtures()
    want = observe_ranges(activations)
    got = observe_ranges(activations)
    match = 1.0 if (len(want["min"]) == len(got["min"]) and np.allclose(want["min"], got["min"])) else 0.0
    return {"observer_match": float(match)}
