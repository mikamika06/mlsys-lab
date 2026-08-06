import numpy as np

np.random.seed(42)

def hadamard_matrix(dim):
    h = np.array([[1.0]], dtype=np.float32)
    n = 1
    while n < dim:
        h = np.block([[h, h], [h, -h]])
        n *= 2
    return h / np.sqrt(dim)

TEST_X = np.random.randn(4, 16).astype(np.float32)
TEST_W = np.random.randn(16, 16).astype(np.float32)

H = hadamard_matrix(16)

def rotate_activation(x):
    return np.matmul(x, H)

def rotate_weight(w):
    return np.matmul(H.T, w)

def compute_fp32_output(x, w):
    x_rot = rotate_activation(x)
    w_rot = rotate_weight(w)
    return np.matmul(x_rot, w_rot)

def baseline_output(x, w):
    return np.matmul(x, w)

def get_outlier_stats(x):
    x_rot = rotate_activation(x)
    return {
        "orig_max": float(np.max(np.abs(x))),
        "rot_max": float(np.max(np.abs(x_rot))),
        "orig_kurtosis": float(np.mean((x - np.mean(x))**4) / (np.std(x)**4 + 1e-8)),
        "rot_kurtosis": float(np.mean((x_rot - np.mean(x_rot))**4) / (np.std(x_rot)**4 + 1e-8))
    }

def rms_norm_weight_fuse(weight, h_matrix):
    return np.matmul(h_matrix.T, weight)
