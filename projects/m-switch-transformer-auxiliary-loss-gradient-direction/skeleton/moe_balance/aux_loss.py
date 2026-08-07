import numpy as np


def compute_switch_aux_loss(logits, alpha=0.01):
    """
    Computes the Switch Transformer auxiliary loss and its analytical gradient
    with respect to router logits.
    """
    raise NotImplementedError


def switch_grad_direction(logits, alpha=0.01):
    """
    Returns the router logit gradient vector dL_aux / dlogits.
    """
    raise NotImplementedError
