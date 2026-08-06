import numpy as np


def generate_test_inputs(seed=42):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 1.0, (2, 8, 64, 64)).astype(np.float32)
    k = rng.normal(0.0, 1.0, (2, 8, 64, 64)).astype(np.float32)
    v = rng.normal(0.0, 1.0, (2, 8, 64, 64)).astype(np.float32)
    q[:, :, :, 10] *= 25.0
    k[:, :, :, 20] *= 30.0
    return q, k, v


def simulate_bf16_attention(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    max_val = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_val)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / sum_exp
    out = np.matmul(attn, v)
    return out.astype(np.float32)


def simulate_fp8_attention_naive(q, k, v):
    q_max = np.max(np.abs(q)) + 1e-5
    k_max = np.max(np.abs(k)) + 1e-5
    v_max = np.max(np.abs(v)) + 1e-5
    q_q = np.clip(np.round(q / q_max * 448.0), -448.0, 448.0) * (q_max / 448.0)
    k_q = np.clip(np.round(k / k_max * 448.0), -448.0, 448.0) * (k_max / 448.0)
    v_q = np.clip(np.round(v / v_max * 448.0), -448.0, 448.0) * (v_max / 448.0)
    return simulate_bf16_attention(q_q, k_q, v_q)


def compute_relative_error(got, want):
    diff = np.linalg.norm(got - want)
    norm = np.linalg.norm(want) + 1e-8
    return float(diff / norm)


def generate_hadamard_matrix(dim):
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32) / np.sqrt(2.0)
    curr = h
    while curr.shape[0] < dim:
        curr = np.kron(curr, h)
    return curr[:dim, :dim]


def apply_hadamard(x):
    dim = x.shape[-1]
    h = generate_hadamard_matrix(dim)
    orig_shape = x.shape
    x_2d = x.reshape(-1, dim)
    transformed = np.matmul(x_2d, h)
    return transformed.reshape(orig_shape)


def measure_incoherence(x):
    max_val = np.max(np.abs(x))
    l2_norm = np.linalg.norm(x) + 1e-8
    dim_prod = np.sqrt(x.size)
    return float(dim_prod * max_val / l2_norm)
