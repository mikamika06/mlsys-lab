import numpy as np

def quantize_q8_0(tensor):
    tensor = np.asarray(tensor, dtype=np.float32)
    shape = tensor.shape
    flat = tensor.reshape(-1, 32)
    max_vals = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.where(max_vals == 0, 1.0, max_vals / 127.0)
    qdata = np.round(flat / scales).astype(np.int8)
    return {"qdata": qdata.reshape(shape), "scales": scales.astype(np.float32)}

def dequantize_q8_0(qdata, scales):
    shape = qdata.shape
    flat_q = qdata.reshape(-1, 32)
    deq = flat_q.astype(np.float32) * scales
    return deq.reshape(shape)

def quantize_q4_0(tensor):
    tensor = np.asarray(tensor, dtype=np.float32)
    shape = tensor.shape
    flat = tensor.reshape(-1, 32)
    max_vals = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.where(max_vals == 0, 1.0, max_vals / 7.0)
    qdata = np.clip(np.round(flat / scales), -8, 7).astype(np.int8)
    return {"qdata": qdata.reshape(shape), "scales": scales.astype(np.float32)}

def dequantize_q4_0(qdata, scales):
    shape = qdata.shape
    flat_q = qdata.reshape(-1, 32)
    deq = flat_q.astype(np.float32) * scales
    return deq.reshape(shape)

def compute_imatrix(activations):
    acts = np.asarray(activations, dtype=np.float32)
    if acts.ndim > 2:
        acts = acts.reshape(-1, acts.shape[-1])
    imat = np.mean(acts ** 2, axis=0)
    return imat

def quantize_imatrix(tensor, imatrix, n_bits=4):
    tensor = np.asarray(tensor, dtype=np.float32)
    shape = tensor.shape
    flat = tensor.reshape(-1, 32)
    imat_flat = np.resize(imatrix, flat.shape[1])
    weights = np.sqrt(imat_flat + 1e-6)

    weighted_flat = flat * weights
    max_val = np.max(np.abs(weighted_flat), axis=1, keepdims=True)
    max_q = (1 << (n_bits - 1)) - 1
    scales = np.where(max_val == 0, 1.0, max_val / max_q)

    qdata = np.clip(np.round(weighted_flat / scales), -max_q, max_q).astype(np.int8)
    deq = (qdata.astype(np.float32) * scales) / weights
    return {"qdata": qdata.reshape(shape), "scales": scales.astype(np.float32), "dequantized": deq.reshape(shape)}
