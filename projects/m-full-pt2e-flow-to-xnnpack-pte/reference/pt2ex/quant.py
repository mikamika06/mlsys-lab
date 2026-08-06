import numpy as np


def observe_ranges(activations):
    mins = [float(np.min(a)) for a in activations]
    maxs = [float(np.max(a)) for a in activations]
    return {"min": mins, "max": maxs}


def compute_qparams(tensor, per_channel=False, axis=0):
    if per_channel:
        scales = []
        zeros = []
        slices = np.moveaxis(tensor, axis, 0)
        for s in slices:
            mn = float(np.min(s))
            mx = float(np.max(s))
            mn = min(mn, 0.0)
            mx = max(mx, 0.0)
            qmin, qmax = -128.0, 127.0
            scale = max((mx - mn) / (qmax - qmin), 1e-8)
            zero = round(-mn / scale + qmin)
            zero = int(np.clip(zero, qmin, qmax))
            scales.append(scale)
            zeros.append(zero)
        return {"scale": np.array(scales, dtype=np.float32), "zero_point": np.array(zeros, dtype=np.int32), "per_channel": True}
    else:
        mn = float(np.min(tensor))
        mx = float(np.max(tensor))
        mn = min(mn, 0.0)
        mx = max(mx, 0.0)
        qmin, qmax = -128.0, 127.0
        scale = max((mx - mn) / (qmax - qmin), 1e-8)
        zero = round(-mn / scale + qmin)
        zero = int(np.clip(zero, qmin, qmax))
        return {"scale": float(scale), "zero_point": int(zero), "per_channel": False}


def convert_tensor(tensor, qparams):
    if qparams["per_channel"]:
        axis = 0
        slices = np.moveaxis(tensor, axis, 0)
        quantized_slices = []
        for i, s in enumerate(slices):
            sc = qparams["scale"][i]
            zp = qparams["zero_point"][i]
            q = np.clip(np.round(s / sc) + zp, -128, 127).astype(np.int8)
            deq = (q.astype(np.float32) - zp) * sc
            quantized_slices.append(deq)
        reconstructed = np.moveaxis(np.stack(quantized_slices, axis=0), 0, axis)
        return reconstructed
    else:
        sc = qparams["scale"]
        zp = qparams["zero_point"]
        q = np.clip(np.round(tensor / sc) + zp, -128, 127).astype(np.int8)
        deq = (q.astype(np.float32) - zp) * sc
        return deq
