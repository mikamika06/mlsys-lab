import numpy as np


def run_eval(model, x, y):
    preds = model.forward(x)
    mse = float(np.mean((preds - y) ** 2))
    accuracy = float(np.mean(np.abs(preds - y) < 0.5))
    return {"mse": mse, "accuracy": accuracy}
