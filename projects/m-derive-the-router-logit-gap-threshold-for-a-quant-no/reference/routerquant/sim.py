import numpy as np


def simulate_argmax_flip(logits, threshold):
    perturbation = np.random.uniform(-threshold, threshold, size=logits.shape)
    perturbed_logits = logits + perturbation
    original_argmax = np.argmax(logits, axis=-1)
    new_argmax = np.argmax(perturbed_logits, axis=-1)
    flips = original_argmax != new_argmax
    return {"flip_rate": float(np.mean(flips)), "flips": flips}
