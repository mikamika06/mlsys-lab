import sys
import numpy as np

sys.path.insert(0, ".")
from quant.eval import Evaluator
from quant.sensitivity import compute_layer_sensitivity, select_mixed_precision_config


class DummyModel:
    """Dummy model fixture for testing regression checks."""

    def __init__(self):
        np.random.seed(42)
        self.layers = {
            "layer1": np.random.randn(10, 10).astype(np.float32) * 2.0,
            "layer2": np.random.randn(10, 10).astype(np.float32) * 0.1,
            "layer3": np.random.randn(10, 10).astype(np.float32) * 0.05,
            "layer4": np.random.randn(10, 10).astype(np.float32) * 0.01,
        }

    def forward(self, x):
        h = x
        for _, w in self.layers.items():
            h = np.dot(h, w)
        return h


def test_evaluation_baseline():
    """Ensures baseline model evaluation runs accurately within [0, 1]."""
    dataset = [(np.ones((1, 10), dtype=np.float32), 0) for _ in range(5)]
    model = DummyModel()
    evaluator = Evaluator(dataset)
    acc = evaluator.evaluate(model)
    assert 0.0 <= acc <= 1.0


def test_mixed_precision_fallback():
    """Ensures sensitivity calculations correctly produce a valid mixed-precision config."""
    dataset = [(np.ones((1, 10), dtype=np.float32), 0) for _ in range(5)]
    model = DummyModel()
    evaluator = Evaluator(dataset)
    sens = compute_layer_sensitivity(model, evaluator, [])
    config = select_mixed_precision_config(sens)
    assert isinstance(config, dict)
    assert len(config) == len(model.layers)
    assert "layer1" in config
    assert config["layer1"] in (4, 8)
