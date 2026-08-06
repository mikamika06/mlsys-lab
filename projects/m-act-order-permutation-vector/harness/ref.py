import numpy as np

def act_order_perm(H):
    return np.argsort(-np.diag(H))

def find_damping(H, start_damp=1e-5, step=10.0, max_iter=10):
    diag = np.diag(H)
    mean_diag = np.mean(diag)
    for i in range(max_iter):
        damp = start_damp * (step ** i)
        H_damp = H.copy()
        np.fill_diagonal(H_damp, diag + damp * mean_diag)
        try:
            np.linalg.cholesky(H_damp)
            return damp
        except np.linalg.LinAlgError:
            pass
    raise ValueError("Cholesky failed")

def lazy_batch_update(W, H_inv, errors, block_start, block_size):
    col_end = block_start + block_size
    if col_end < W.shape[1]:
        W[:, col_end:] -= errors @ H_inv[block_start:col_end, col_end:]
    return W

def make_fixtures():
    np.random.seed(42)
    fixtures = []
    for _ in range(5):
        X = np.random.randn(100, 64)
        H = X.T @ X
        H[:, -1] = H[:, 0]
        H[-1, :] = H[0, :]
        H[:, -2] = H[:, 1]
        H[-2, :] = H[1, :]
        
        W = np.random.randn(128, 64)
        errors = np.random.randn(128, 16)
        
        fixtures.append((H, W, errors))
    return fixtures

FIXTURES = make_fixtures()
