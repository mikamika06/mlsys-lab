import numpy as np

def functional_add(a, b):
    """Return a new array equal to a + b."""
    out = np.empty_like(a)
    for idx in np.ndindex(a.shape):
        out[idx] = a[idx] + b[idx]
    return out

def functional_relu(x):
    """Return a new array with ReLU applied elementwise."""
    out = np.empty_like(x)
    for idx in np.ndindex(x.shape):
        val = x[idx]
        if val > 0:
            out[idx] = val
        else:
            out[idx] = 0
    return out

def functional_copy(a):
    """Return a copy of the input array."""
    out = np.empty_like(a)
    for idx in np.ndindex(a.shape):
        out[idx] = a[idx]
    return out
