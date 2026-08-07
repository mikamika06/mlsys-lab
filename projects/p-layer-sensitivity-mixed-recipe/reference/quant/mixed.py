import numpy as np

def apply_mixed_quantization(model, recipe):
    quantized = {}
    for name, layer in model.items():
        bits = recipe.get(name, 4)
        scale = 2.0 ** (bits - 1)
        quantized[name] = np.round(layer * scale) / scale
    return quantized

def evaluate_model(model, dataloader):
    losses = []
    for x, y in dataloader:
        pred = x
        for name in sorted(model.keys()):
            pred = np.dot(pred, model[name])
        loss = np.mean((pred - y) ** 2)
        losses.append(loss)
    return float(np.mean(losses))
