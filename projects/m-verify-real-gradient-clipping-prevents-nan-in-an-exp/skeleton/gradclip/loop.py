import numpy as np
from gradclip.clipping import clip_grad_norm


def run_training_step(weights, grad, max_norm, lr):
    raise NotImplementedError
