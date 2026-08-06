import numpy as np

np.random.seed(1337)
MATRIX_CHAIN = [np.random.randn(16, 16).astype(np.float32) for _ in range(4)]
LAYER_INPUT = np.random.randn(8, 16).astype(np.float32) * 5.0

LAYERS = [
    {"type": "linear", "w": np.random.randn(32, 16).astype(np.float32), "b": np.random.randn(32).astype(np.float32)},
    {"type": "layernorm", "gamma": np.ones(16, dtype=np.float32), "beta": np.zeros(16, dtype=np.float32), "eps": 1e-5},
    {"type": "relu"},
    {"type": "layernorm", "gamma": np.random.randn(16).astype(np.float32) * 10, "beta": np.random.randn(16).astype(np.float32), "eps": 1e-5}
]

def truncate_to_tf32(x: np.ndarray) -> np.ndarray:
    x_32 = np.ascontiguousarray(x, dtype=np.float32)
    return (x_32.view(np.uint32) & 0xFFFFE000).view(np.float32)

def truncate_to_bf16(x: np.ndarray) -> np.ndarray:
    x_32 = np.ascontiguousarray(x, dtype=np.float32)
    return (x_32.view(np.uint32) & 0xFFFF0000).view(np.float32)

def matmul_chain(matrices: list[np.ndarray], precision: str) -> np.ndarray:
    if not matrices:
        return np.array([])
    res = matrices[0]
    for m in matrices[1:]:
        a_mat = res
        b_mat = m
        if precision == 'tf32':
            a_mat = truncate_to_tf32(a_mat)
            b_mat = truncate_to_tf32(b_mat)
        elif precision == 'bf16':
            a_mat = truncate_to_bf16(a_mat)
            b_mat = truncate_to_bf16(b_mat)
        res = a_mat @ b_mat
    return res

def find_unsafe_layers(layers: list[dict], x: np.ndarray, threshold: float) -> list[int]:
    unsafe = []
    def cast(v):
        if isinstance(v, np.ndarray):
            return truncate_to_bf16(v)
        return v

    for i, layer in enumerate(layers):
        if layer["type"] == "linear":
            out_fp = x @ layer["w"].T + layer["b"]
        elif layer["type"] == "layernorm":
            mu = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            out_fp = (x - mu) / np.sqrt(var + layer["eps"]) * layer["gamma"] + layer["beta"]
        elif layer["type"] == "relu":
            out_fp = np.maximum(0, x)
        else:
            continue

        x_c = cast(x)
        if layer["type"] == "linear":
            out_bf = x_c @ cast(layer["w"]).T + cast(layer["b"])
        elif layer["type"] == "layernorm":
            mu = np.mean(x_c, axis=-1, keepdims=True)
            var = np.var(x_c, axis=-1, keepdims=True)
            out_bf = (x_c - mu) / np.sqrt(var + layer["eps"]) * cast(layer["gamma"]) + cast(layer["beta"])
        elif layer["type"] == "relu":
            out_bf = np.maximum(0, x_c)

        rel_err = np.linalg.norm(out_bf - out_fp) / (np.linalg.norm(out_fp) + 1e-12)
        if rel_err > threshold:
            unsafe.append(i)

    return unsafe
