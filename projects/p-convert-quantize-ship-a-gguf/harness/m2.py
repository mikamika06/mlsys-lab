import sys
import numpy as np
import ref

def check(workdir):
    out = {"q4_0_size_ratio": 1.0, "q8_0_size_ratio": 1.0, "quant_reconstruction_ok": 0.0}

    sys.path.insert(0, workdir)
    from gguf_pipeline.quantizer import quantize_q4_0, quantize_q8_0, dequantize_q8_0

    np.random.seed(42)
    x = np.random.randn(64, 64).astype(np.float32)

    q4 = quantize_q4_0(x)
    q8 = quantize_q8_0(x)

    orig_bytes = x.nbytes
    q4_bytes = q4["qdata"].nbytes + q4["scales"].nbytes
    q8_bytes = q8["qdata"].nbytes + q8["scales"].nbytes

    out["q4_0_size_ratio"] = float(q4_bytes / orig_bytes)
    out["q8_0_size_ratio"] = float(q8_bytes / orig_bytes)

    deq8 = dequantize_q8_0(q8["qdata"], q8["scales"])
    err = np.mean(np.abs(x - deq8))
    if err < 0.15:
        out["quant_reconstruction_ok"] = 1.0

    return out
