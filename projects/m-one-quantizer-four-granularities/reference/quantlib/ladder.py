import numpy as np
from reference.quantlib.core import dequantize, quantize
from reference.quantlib.params import compute_qparams


def evaluate_ladder(tensor, bits=8):
    granularities = ["tensor", "channel", "group", "block"]
    results = []
    for g in granularities:
        gs = 32 if g in ("group", "block") else 128
        scales, zps = compute_qparams(tensor, bits=bits, symmetric=True, granularity=g, group_size=gs)
        if g == "tensor":
            q = quantize(tensor, (scales[0], zps[0]), bits=bits, symmetric=True)
            rec = dequantize(q, (scales[0], zps[0]), symmetric=True)
            meta_bytes = 8
        elif g == "channel":
            recs = []
            for i in range(tensor.shape[0]):
                q_sub = quantize(tensor[i], (scales[i], zps[i]), bits=bits, symmetric=True)
                recs.append(dequantize(q_sub, (scales[i], zps[i]), symmetric=True))
            rec = np.stack(recs)
            meta_bytes = len(scales) * 4
        elif g == "group":
            flat = tensor.flatten()
            recs = []
            for idx, (s, z) in enumerate(zip(scales, zps)):
                sub = flat[idx * gs:(idx + 1) * gs]
                q_sub = quantize(sub, (s, z), bits=bits, symmetric=True)
                recs.append(dequantize(q_sub, (s, z), symmetric=True))
            rec = np.concatenate(recs).reshape(tensor.shape)
            meta_bytes = len(scales) * 4
        else:
            h, w = tensor.shape
            bh, bw = gs, gs
            recs = []
            idx = 0
            for r in range(0, h, bh):
                row_recs = []
                for c in range(0, w, bw):
                    sub = tensor[r:r+bh, c:c+bw]
                    s, z = scales[idx], zps[idx]
                    q_sub = quantize(sub, (s, z), bits=bits, symmetric=True)
                    row_recs.append(dequantize(q_sub, (s, z), symmetric=True))
                    idx += 1
                recs.append(np.hstack(row_recs))
            rec = np.vstack(recs)
            meta_bytes = len(scales) * 4
        max_err = float(np.max(np.abs(tensor - rec)))
        results.append({"granularity": g, "max_abs_err": max_err, "meta_bytes": meta_bytes})
    return results
