import numpy as np
import io

def offload_attention(q, k, v):
    # Ensure float64 throughout
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Offload KV to a bytes buffer
    buf = io.BytesIO()
    np.savez_compressed(buf, k=k, v=v)
    buf.seek(0)
    data = np.load(buf, allow_pickle=False)
    k_off = data['k']
    v_off = data['v']

    d_k = q.shape[-1]
    scale = np.sqrt(d_k)

    logits = np.matmul(q, k_off.transpose(0, 2, 1)) / scale
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    out = np.matmul(probs, v_off)

    return out.astype(np.float64)
