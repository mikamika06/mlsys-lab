import numpy as np

def hessian_saliency(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal Hessian salience for each input dimension of a linear layer.
    Parameters
    ----------
    W : ndarray, shape (n_out, n_in)
        Weight matrix of the linear layer.
    A : ndarray, shape (batch_size, n_in)
        Activations that were fed to the layer during inference.
    Returns
    -------
    h : ndarray, shape (n_in,)
        Salience scores for each input dimension.
    """
    return np.sum(A**2, axis=0) * np.sum(W**2, axis=0)
