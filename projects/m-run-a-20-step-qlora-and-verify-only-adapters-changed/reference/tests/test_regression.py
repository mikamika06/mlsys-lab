import sys
import numpy as np

sys.path.insert(0, ".")
from qlora.layer import LinearQLoRA
from qlora.train import train_20_steps


def test_weights_do_not_change():
    layer = LinearQLoRA(16, 32, seed=10)
    X = np.random.normal(0, 1, size=(4, 16)).astype(np.float32)
    target = np.random.normal(0, 1, size=(4, 32)).astype(np.float32)

    orig_weight = layer.weight.copy()
    orig_scale = layer.scale.copy()
    orig_A = layer.lora_A.copy()
    orig_B = layer.lora_B.copy()

    train_20_steps(layer, X, target, lr=0.1)

    assert np.array_equal(layer.weight, orig_weight), "Base weights changed!"
    assert np.array_equal(layer.scale, orig_scale), "Scale changed!"
    assert not np.allclose(layer.lora_A, orig_A), "Adapter A didn't change!"
    assert not np.allclose(layer.lora_B, orig_B), "Adapter B didn't change!"
