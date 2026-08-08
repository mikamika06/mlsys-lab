"""Checkpoint re-sharded loader and forward evaluation."""

import numpy as np
from fsdp_ckpt.converter import from_portable_format, to_portable_format


def reshard_checkpoint(checkpoint_data, target_world_size):
    """Re-shards an arbitrary sharded checkpoint for a new target world size."""
    portable = to_portable_format(checkpoint_data)
    return from_portable_format(portable, target_world_size)


def forward_pass(model_params, x):
    """Run forward inference pass on a 2-layer linear network."""
    w1 = model_params["fc1.weight"]
    b1 = model_params["fc1.bias"]
    w2 = model_params["fc2.weight"]
    b2 = model_params["fc2.bias"]

    h = np.maximum(0, np.dot(x, w1.T) + b1)
    out = np.dot(h, w2.T) + b2
    return out


def compute_loss(model_params, x, y):
    """Compute mean squared error loss."""
    pred = forward_pass(model_params, x)
    return float(np.mean((pred - y) ** 2))
