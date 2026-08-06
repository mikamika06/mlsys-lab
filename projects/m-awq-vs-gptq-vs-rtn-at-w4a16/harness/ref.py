import numpy as np


def generate_fixtures():
    np.random.seed(42)

    X1 = np.random.randn(32, 64)
    X1[:, 5] *= 50.0
    X1[:, 12] *= 100.0
    W1 = np.random.randn(64, 64)

    X2 = np.random.randn(16, 32)
    X2[:, 2] *= 30.0
    W2_a = np.random.randn(32, 32)
    W2_b = np.random.randn(48, 32)

    configs = [
        {
            "layers": [
                {"input": "norm1", "projections": ["attn.q", "attn.k", "attn.v"]},
                {"input": "norm2", "projections": ["mlp.gate", "mlp.up"]},
            ]
        },
        {
            "nodes": [
                {"source": "emb", "branches": ["head.0", "head.1"]},
                {"source": "act1", "branches": ["proj.out"]},
            ]
        },
        {
            "modules": [
                {
                    "in": "block0_in",
                    "targets": ["block0.lin1", "block0.lin2", "block0.lin3"],
                },
                {"in": "block1_in", "targets": ["block1.lin1"]},
            ]
        },
    ]

    return {
        "X1": X1,
        "W1": W1,
        "X2": X2,
        "W2_a": W2_a,
        "W2_b": W2_b,
        "configs": configs,
    }


def quantize_rtn(W, n_bits=4, group_size=32):
    out_features, in_features = W.shape
    W_q = np.zeros_like(W)
    qmax = (1 << n_bits) - 1

    for g in range(0, in_features, group_size):
        w_sub = W[:, g : g + group_size]
        w_min = np.min(w_sub, axis=1, keepdims=True)
        w_max = np.max(w_sub, axis=1, keepdims=True)
        scale = (w_max - w_min) / float(qmax) + 1e-8
        zero = np.round(-w_min / scale)
        q = np.clip(np.round(w_sub / scale) + zero, 0, qmax)
        W_q[:, g : g + group_size] = (q - zero) * scale

    return W_q


def quantize_gptq_sim(W, X, n_bits=4, group_size=32):
    out_features, in_features = W.shape
    H = 2.0 * np.dot(X.T, X) / float(X.shape[0]) + 1e-4 * np.eye(in_features)
    H_inv = np.linalg.inv(H)

    W_q = W.copy()
    qmax = (1 << n_bits) - 1

    for j in range(in_features):
        col = W_q[:, j : j + 1]
        c_min = np.min(col)
        c_max = np.max(col)
        scale = (c_max - c_min) / float(qmax) + 1e-8
        zero = np.round(-c_min / scale)
        q_col = np.clip(np.round(col / scale) + zero, 0, qmax)
        deq_col = (q_col - zero) * scale

        err = col - deq_col
        W_q[:, j : j + 1] = deq_col

        if j + 1 < in_features:
            weights_update = H_inv[j, j + 1 :] / H_inv[j, j]
            W_q[:, j + 1 :] -= np.outer(err.squeeze(), weights_update)

    return W_q


def quantize_awq_sim(W, X, n_bits=4, group_size=32, max_scale_ratio=5.0):
    Sx = np.mean(np.abs(X), axis=0)
    Sw = np.mean(np.abs(W), axis=0)

    best_mse = float("inf")
    best_Wq = None

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s_raw = (Sx**alpha) / (Sw ** (1.0 - alpha) + 1e-8)
        s_norm = s_raw / np.mean(s_raw)
        s_cap = max_scale_ratio * np.median(s_norm)
        s_floor = 1.0 / max_scale_ratio
        s = np.clip(s_norm, s_floor, s_cap)

        W_scaled = W * s[None, :]
        W_q_scaled = quantize_rtn(W_scaled, n_bits=n_bits, group_size=group_size)
        W_q = W_q_scaled / s[None, :]

        Y = np.dot(X, W.T)
        Y_hat = np.dot(X, W_q.T)
        mse = np.mean((Y - Y_hat) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_Wq = W_q

    return best_Wq


def compare_methods(W, X, n_bits=4, group_size=32):
    Y = np.dot(X, W.T)

    W_rtn = quantize_rtn(W, n_bits=n_bits, group_size=group_size)
    mse_rtn = float(np.mean((Y - np.dot(X, W_rtn.T)) ** 2))

    W_gptq = quantize_gptq_sim(W, X, n_bits=n_bits, group_size=group_size)
    mse_gptq = float(np.mean((Y - np.dot(X, W_gptq.T)) ** 2))

    W_awq = quantize_awq_sim(W, X, n_bits=n_bits, group_size=group_size)
    mse_awq = float(np.mean((Y - np.dot(X, W_awq.T)) ** 2))

    return {"rtn_mse": mse_rtn, "gptq_mse": mse_gptq, "awq_mse": mse_awq}


def derive_awq_mappings(architecture_config):
    mapping = {}
    items = architecture_config.get(
        "layers",
        architecture_config.get("nodes", architecture_config.get("modules", [])),
    )
    for item in items:
        src = item.get("input", item.get("source", item.get("in")))
        targets = item.get("projections", item.get("branches", item.get("targets")))
        if src and targets:
            mapping[src] = sorted(list(targets))
    return mapping


def compute_awq_scales(W_list, X, alpha=0.5, max_scale_ratio=5.0):
    Sx = np.mean(np.abs(X), axis=0)
    Sw_list = [np.mean(np.abs(W), axis=0) for W in W_list]
    Sw = np.mean(Sw_list, axis=0)

    s_raw = (Sx**alpha) / (Sw ** (1.0 - alpha) + 1e-8)
    s_norm = s_raw / np.mean(s_raw)

    s_cap = max_scale_ratio * np.median(s_norm)
    s_floor = 1.0 / max_scale_ratio
    return np.clip(s_norm, s_floor, s_cap)
