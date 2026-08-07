import sys
import numpy as np

sys.path.insert(0, ".")
from compress.checkpoint import load_checkpoint, save_checkpoint
from compress.transform import round_trip


def test_round_trip_preserves_shape():
    data = {"weight": np.random.randn(16, 16).astype(np.float32)}
    res = round_trip(data)
    assert res["weight"].shape == data["weight"].shape


def test_round_trip_max_error():
    data = {"weight": np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)}
    res = round_trip(data)
    err = np.max(np.abs(res["weight"] - data["weight"]))
    assert err <= 1e-5
