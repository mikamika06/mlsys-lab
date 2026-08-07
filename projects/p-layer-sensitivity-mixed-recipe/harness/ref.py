import numpy as np
from reference.quant.sensitivity import measure_sensitivity
from reference.quant.recipe import build_recipe
from reference.quant.mixed import apply_mixed_quantization, evaluate_model

def get_dummy_data():
    np.random.seed(42)
    model = {
        "layer_0": np.random.randn(4, 4),
        "layer_1": np.random.randn(4, 4),
        "layer_2": np.random.randn(4, 4),
        "layer_3": np.random.randn(4, 4)
    }
    dataloader = [(np.random.randn(2, 4), np.random.randn(2, 4)) for _ in range(3)]
    return model, dataloader
