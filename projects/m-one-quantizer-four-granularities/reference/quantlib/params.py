import numpy as np


def compute_qparams(tensor, bits=8, symmetric=True, granularity="tensor", group_size=128):
    if granularity == "tensor":
        if symmetric:
            max_val = np.max(np.abs(tensor))
            qmax = float((1 << (bits - 1)) - 1)
            scale = max_val / qmax if max_val > 0 else 1.0
            return np.array([scale], dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            min_val, max_val = np.min(tensor), np.max(tensor)
            qmax = float((1 << bits) - 1)
            scale = (max_val - min_val) / qmax if max_val > min_val else 1.0
            zp = np.round(-min_val / scale) if scale > 0 else 0.0
            return np.array([scale], dtype=np.float32), np.array([zp], dtype=np.float32)
    elif granularity == "channel":
        scales = []
        zps = []
        for i in range(tensor.shape[0]):
            sub = tensor[i]
            if symmetric:
                max_val = np.max(np.abs(sub))
                qmax = float((1 << (bits - 1)) - 1)
                scale = max_val / qmax if max_val > 0 else 1.0
                scales.append(scale)
                zps.append(0.0)
            else:
                min_val, max_val = np.min(sub), np.max(sub)
                qmax = float((1 << bits) - 1)
                scale = (max_val - min_val) / qmax if max_val > min_val else 1.0
                zp = np.round(-min_val / scale) if scale > 0 else 0.0
                scales.append(scale)
                zps.append(zp)
        return np.array(scales, dtype=np.float32), np.array(zps, dtype=np.float32)
    elif granularity == "group":
        flat = tensor.flatten()
        n = len(flat)
        scales = []
        zps = []
        for i in range(0, n, group_size):
            sub = flat[i:i + group_size]
            if symmetric:
                max_val = np.max(np.abs(sub))
                qmax = float((1 << (bits - 1)) - 1)
                scale = max_val / qmax if max_val > 0 else 1.0
                scales.append(scale)
                zps.append(0.0)
            else:
                min_val, max_val = np.min(sub), np.max(sub)
                qmax = float((1 << bits) - 1)
                scale = (max_val - min_val) / qmax if max_val > min_val else 1.0
                zp = np.round(-min_val / scale) if scale > 0 else 0.0
                scales.append(scale)
                zps.append(zp)
        return np.array(scales, dtype=np.float32), np.array(zps, dtype=np.float32)
    elif granularity == "block":
        h, w = tensor.shape
        bh, bw = group_size, group_size
        scales = []
        zps = []
        for r in range(0, h, bh):
            for c in range(0, w, bw):
                sub = tensor[r:r+bh, c:c+bw]
                if symmetric:
                    max_val = np.max(np.abs(sub))
                    qmax = float((1 << (bits - 1)) - 1)
                    scale = max_val / qmax if max_val > 0 else 1.0
                    scales.append(scale)
                    zps.append(0.0)
                else:
                    min_val, max_val = np.min(sub), np.max(sub)
                    qmax = float((1 << bits) - 1)
                    scale = (max_val - min_val) / qmax if max_val > min_val else 1.0
                    zp = np.round(-min_val / scale) if scale > 0 else 0.0
                    scales.append(scale)
                    zps.append(zp)
        return np.array(scales, dtype=np.float32), np.array(zps, dtype=np.float32)
    else:
        raise ValueError("unknown granularity")
