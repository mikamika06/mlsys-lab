import numpy as np


def generate_tensor():
    np.random.seed(42)
    t1 = np.random.normal(loc=-2.0, scale=0.5, size=2048)
    t2 = np.random.normal(loc=2.0, scale=0.5, size=2048)
    t = np.concatenate([t1, t2])
    np.random.shuffle(t)
    return t.astype(np.float32).reshape(64, 64)


def palettize_scalar(tensor, bits, iters):
    K = 2 ** bits
    palette = np.linspace(tensor.min(), tensor.max(), K).astype(np.float32)
    flat = tensor.ravel()

    indices = np.zeros_like(flat, dtype=np.int32)
    for _ in range(iters):
        dist = (flat[:, None] - palette[None, :]) ** 2
        indices = np.argmin(dist, axis=1)
        for i in range(K):
            mask = (indices == i)
            if np.any(mask):
                palette[i] = flat[mask].mean()

    return palette, indices.reshape(tensor.shape)


def palettize_vector(tensor, bits, block_size, iters):
    K = 2 ** bits
    vecs = tensor.ravel().reshape(-1, block_size)
    N_vec = vecs.shape[0]

    init_indices = np.linspace(0, N_vec - 1, K).astype(int)
    palette = vecs[init_indices].copy().astype(np.float32)

    indices = np.zeros(N_vec, dtype=np.int32)
    for _ in range(iters):
        dist = np.sum((vecs[:, None, :] - palette[None, :, :]) ** 2, axis=2)
        indices = np.argmin(dist, axis=1)
        for i in range(K):
            mask = (indices == i)
            if np.any(mask):
                palette[i] = vecs[mask].mean(axis=0)

    return palette, indices


def palettize_size_bytes(num_elements, bits, block_size=1):
    num_indices = num_elements // block_size
    indices_bytes = (num_indices * bits + 7) // 8
    K = 2 ** bits
    palette_bytes = K * block_size * 4
    return indices_bytes + palette_bytes
