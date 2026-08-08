import numpy as np


def palettize_scalar(tensor, bits, iters):
    """
    K-means palettization for a scalar tensor.
    Initialize the palette with np.linspace(tensor.min(), tensor.max(), 2**bits).
    If a cluster becomes empty, keep its old centroid.
    Return (palette, indices) where palette is float32 and indices is int32 matching the tensor shape.
    """
    raise NotImplementedError


def palettize_vector(tensor, bits, block_size, iters):
    """
    Vector k-means palettization.
    Group elements of the flattened tensor into blocks of `block_size`.
    Initialize the palette with vecs[np.linspace(0, vecs.shape[0]-1, 2**bits).astype(int)].
    Return (palette, indices).
    """
    raise NotImplementedError


def palettize_size_bytes(num_elements, bits, block_size=1):
    """
    Calculate the exact total byte size for the palettized tensor.
    The indices are bit-packed (round up to nearest byte).
    The float32 palette must also be stored.
    """
    raise NotImplementedError
