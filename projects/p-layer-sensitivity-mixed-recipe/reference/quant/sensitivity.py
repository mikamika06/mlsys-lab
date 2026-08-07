import numpy as np

def measure_sensitivity(model, dataloader):
    sensitivities = {}
    for name, layer in model.items():
        errs = []
        for x, y in dataloader:
            orig = np.dot(x, layer)
            q_layer = np.round(layer * 2) / 2
            q_out = np.dot(x, q_layer)
            err = np.mean((orig - q_out) ** 2)
            errs.append(err)
        sensitivities[name] = float(np.mean(errs))
    return sensitivities
