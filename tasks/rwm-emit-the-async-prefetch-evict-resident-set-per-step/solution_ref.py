from typing import List, Set

def emit_resident_set_per_step(num_layers: int, num_steps: int) -> List[Set[int]]:
    """
    Reference implementation of the async prefetch/evict scheduler.
    Returns a list of resident‑set snapshots at the start of each decoding step.
    """
    res: List[Set[int]] = []
    if num_layers == 0:
        return res

    resident: Set[int] = set()
    for step in range(num_steps):
        cur = step % num_layers
        if step == 0:
            # initial resident set contains layer 0 and, if present, layer 1
            resident.add(cur)
            if num_layers > 1:
                resident.add((cur + 1) % num_layers)

        # Emit the resident set at the start of this step
        res.append(set(resident))

        # Evict the current layer after it has been used
        resident.discard(cur)

        # Prefetch the layer two steps ahead for the next step
        nxt = (cur + 2) % num_layers
        resident.add(nxt)

    return res
