import numpy as np


def dense_vs_compressed24_matmul_error(W: np.ndarray, X: np.ndarray) -> float:
    """W is (m,n), already exactly 2:4 sparse (2 nonzeros per group of 4
    columns), n % 4 == 0. X is (n,p). Compute Y_dense = W @ X. Separately,
    for every row and group of 4 columns, read the 2 nonzero values (left-
    to-right) and their in-group position (0..3), scatter them into a
    fresh zero (m,n) buffer, matmul with X to get Y_compressed. Return
    float(max(|Y_dense - Y_compressed|))."""
    raise NotImplementedError('your code here')
