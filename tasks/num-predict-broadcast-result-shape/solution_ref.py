import numpy as np

def broadcast_shape(shape1, shape2):
    """
    Return the NumPy broadcast result shape of two input shapes.
    If the shapes are not compatible, return an empty tuple ().
    """
    try:
        return tuple(np.broadcast_shapes(shape1, shape2))
    except ValueError:
        return ()
