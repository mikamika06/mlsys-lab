import numpy as np


def create_batch_mapping(active_requests, req_to_cache_slot, max_cache_batch):
    """Map active request IDs in current batch order to physical cache batch slots."""
    out = []
    seen = set()
    for req in active_requests:
        if req not in req_to_cache_slot:
            raise ValueError(f"Request {req} has no assigned cache slot")
        slot = req_to_cache_slot[req]
        if not (0 <= slot < max_cache_batch):
            raise ValueError(f"Slot {slot} out of bounds for max_cache_batch {max_cache_batch}")
        if slot in seen:
            raise ValueError(f"Duplicate slot assignment {slot}")
        seen.add(slot)
        out.append(slot)
    return np.array(out, dtype=np.int32)


def remap_batch_indices(old_cache_batch_idx, active_mask):
    """Remap cache batch indices when dropping finished requests."""
    old_cache_batch_idx = np.asarray(old_cache_batch_idx, dtype=np.int32)
    active_mask = np.asarray(active_mask, dtype=bool)
    if len(old_cache_batch_idx) != len(active_mask):
        raise ValueError("Length mismatch between old_cache_batch_idx and active_mask")
    return old_cache_batch_idx[active_mask]
