import numpy as np

class BundledProgram:
    def __init__(self, weights, diverge_layer=-1):
        self.weights = weights
        self.diverge_layer = diverge_layer

    def run_exported(self, x):
        outs = []
        curr = x
        for i, w in enumerate(self.weights):
            if self.diverge_layer != -1 and i >= self.diverge_layer:
                curr = np.tanh(np.dot(curr, w)) + 0.4 * np.ones_like(curr)
            else:
                curr = np.tanh(np.dot(curr, w))
            outs.append(curr)
        return outs

def run_eager(weights, x):
    outs = []
    curr = x
    for w in weights:
        curr = np.tanh(np.dot(curr, w))
        outs.append(curr)
    return outs
