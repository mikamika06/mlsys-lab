import numpy as np


class Evaluator:
    """Evaluates model accuracy against a target dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self, model):
        correct = 0
        total = len(self.dataset)
        for x, target in self.dataset:
            logits = model.forward(x)
            pred = int(np.argmax(logits))
            if pred == int(target):
                correct += 1
        return correct / total if total > 0 else 0.0
