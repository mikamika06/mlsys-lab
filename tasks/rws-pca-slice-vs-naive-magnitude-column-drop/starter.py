import numpy as np


def pca_vs_naive_mse(X: np.ndarray, k: int):
    """
    Return (mse_pca, mse_naive): reconstruction MSE of X under (a) a
    rank-k PCA projection (mean-center, top-k SVD components, project +
    reconstruct) and (b) naively zeroing every column except the k
    largest-L2-norm raw columns. See task.md.
    """
    raise NotImplementedError('your code here')
