import numpy as np


def compute_block_influence(model, calibration_data):
    num_layers = model.num_layers
    influences = np.zeros(num_layers)
    for sample in calibration_data:
        _, acts = model.forward(sample, return_activations=True)
        for i in range(num_layers):
            inp = acts[i]
            outp = acts[i + 1]
            diff = outp - inp
            cos_sim = np.sum(diff * inp) / (np.linalg.norm(diff) * np.linalg.norm(inp) + 1e-8)
            influences[i] += np.abs(1.0 - cos_sim)
    influences /= len(calibration_data)
    return influences
