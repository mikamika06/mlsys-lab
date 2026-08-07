import numpy as np

def generate_matmul_fixtures():
    np.random.seed(42)
    A = np.random.randn(16, 32).astype(np.float32)
    B = np.random.randn(32, 16).astype(np.float32)
    return A, B

def per_block_int8_matmul(A, B, block_size):
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    for k in range(0, K, block_size):
        A_b = A[:, k:k+block_size]
        B_b = B[k:k+block_size, :]

        sA = np.max(np.abs(A_b), axis=1, keepdims=True) / 127.0
        sA = np.clip(sA, 1e-9, None)
        Aq = np.round(A_b / sA)

        sB = np.max(np.abs(B_b), axis=0, keepdims=True) / 127.0
        sB = np.clip(sB, 1e-9, None)
        Bq = np.round(B_b / sB)

        C += np.dot(Aq, Bq) * (sA @ sB)
    return C

def generate_quant_fixtures():
    np.random.seed(43)
    return np.random.uniform(0.1, 400.0, size=(1000,)).astype(np.float32)

def simulate_e4m3(x):
    s = np.sign(x)
    v = np.abs(x)
    v = np.clip(v, 2**-6, 448.0)
    e = np.floor(np.log2(v))
    m = np.round((v / (2**e) - 1.0) * 8.0) / 8.0
    return s * (2**e) * (1.0 + m)

def e4m3_max_rel_error(x):
    xq = simulate_e4m3(x)
    errs = np.abs(x - xq) / (np.abs(x) + 1e-9)
    return float(np.max(errs))

def generate_scaling_fixtures():
    np.random.seed(44)
    x = np.random.randn(128).astype(np.float32)
    x[0] = 5000.0
    return x

def per_tensor_vs_block(x, block_size):
    x = x.flatten()
    scale_t = max(float(np.max(np.abs(x))) / 127.0, 1e-9)
    xq_t = np.round(x / scale_t) * scale_t
    err_tensor = float(np.mean((x - xq_t)**2))

    err_block_sum = 0.0
    for i in range(0, len(x), block_size):
        block = x[i:i+block_size]
        scale_b = max(float(np.max(np.abs(block))) / 127.0, 1e-9)
        bq = np.round(block / scale_b) * scale_b
        err_block_sum += np.sum((block - bq)**2)

    err_block = float(err_block_sum / len(x))
    return err_tensor, err_block
