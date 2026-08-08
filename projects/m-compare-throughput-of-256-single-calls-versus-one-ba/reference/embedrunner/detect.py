import numpy as np

def is_l2_normalized(embeddings, tol=1e-5) -> bool:
    arr = np.array(embeddings)
    if arr.ndim == 1:
        arr = arr.expand_dims(0) if hasattr(arr, "expand_dims") else np.expand_dims(arr, 0)
    norms = np.linalg.norm(arr, axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))
