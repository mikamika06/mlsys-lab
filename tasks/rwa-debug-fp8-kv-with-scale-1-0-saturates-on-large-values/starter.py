import numpy as np


def _fake_fp8_roundtrip(x):
    # TODO: this intentionally keeps scale=1.0 and saturates large values.
    return np.clip(np.asarray(x, dtype=np.float64), -448.0, 448.0)


def fp8_attention_output(Q, K, V):
    K_hat = _fake_fp8_roundtrip(K)
    V_hat = _fake_fp8_roundtrip(V)
    scores = Q @ K_hat.T / np.sqrt(Q.shape[1])
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs @ V_hat
