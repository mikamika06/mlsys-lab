import numpy as np


def create_batch_mapping(active_requests, req_to_cache_slot, max_cache_batch):
    """Map active request IDs in current batch order to physical cache batch slots."""
    raise NotImplementedError


def remap_batch_indices(old_cache_batch_idx, active_mask):
    """Remap cache batch indices when dropping finished requests."""
    raise NotImplementedError
