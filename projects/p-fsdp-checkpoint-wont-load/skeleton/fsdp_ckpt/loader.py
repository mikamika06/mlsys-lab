"""Checkpoint re-sharded loader and forward evaluation."""


def reshard_checkpoint(checkpoint_data, target_world_size):
    """Re-shards an arbitrary sharded checkpoint for a new target world size."""
    raise NotImplementedError


def forward_pass(model_params, x):
    """Run forward inference pass on a 2-layer linear network."""
    raise NotImplementedError


def compute_loss(model_params, x, y):
    """Compute mean squared error loss."""
    raise NotImplementedError
