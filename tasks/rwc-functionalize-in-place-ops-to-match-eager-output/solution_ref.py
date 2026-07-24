import numpy as np

def functional_add(a, b):
    """Return a new array equal to a + b."""
    return a + b

def functional_relu(x):
    """Return a new array with ReLU applied elementwise."""
    return np.maximum(0, x)

def functional_copy(a):
    """Return a copy of the input array."""
    return a.copy()
