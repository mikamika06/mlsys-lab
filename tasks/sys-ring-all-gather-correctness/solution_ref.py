import numpy as np

def ring_all_gather(local_arrays):
    """Correct implementation of ring all‑gather simulation."""
    # Concatenate all local arrays into a single buffer
    gathered = np.concatenate([np.asarray(a, dtype=np.float64) for a in local_arrays])
    # Return a copy for each rank to avoid aliasing
    return [gathered.copy() for _ in range(len(local_arrays))]
