import numpy as np

def select_quant_types(weights, imatrix, threshold=1.0):
    choices = {}
    for name, w in weights.items():
        imp = np.mean(imatrix[name]) if name in imatrix else 1.0
        if imp > threshold:
            choices[name] = "Q8_0"
        else:
            choices[name] = "Q4_0"
    return choices
