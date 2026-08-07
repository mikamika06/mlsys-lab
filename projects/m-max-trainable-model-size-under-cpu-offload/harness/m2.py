import numpy as np
import ref

def check(workdir):
    from offload.quantize import int8_block_dequantize, int8_block_quantize
    out = {"quant_match": 0.0, "rel_err": 1.0}
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((128, 256)).astype(np.float32)
    block_size = 64
    try:
        q, scales = int8_block_quantize(arr, block_size)
        ref_q, ref_scales = ref.int8_block_quantize(arr, block_size)
        recon = int8_block_dequantize(q, scales, arr.shape)
        err = float(np.linalg.norm(arr - recon) / np.linalg.norm(arr))
        out["rel_err"] = err
        if q.dtype == np.int8 and q.shape == ref_q.shape and np.allclose(scales, ref_scales, rtol=1e-4):
            out["quant_match"] = 1.0
        else:
            out["_note"] = f"mismatch in quantization structure or dtype"
    except Exception as e:
        out["_note"] = f"quantization failed: {type(e).__name__}: {str(e)[:120]}"
    return out
