import ref
import numpy as np


def check(workdir):
    out = {"byte_exact_fraction": 0.0, "dequantize_match": 0.0, "compare_runs": 0.0}

    np.random.seed(42)
    tensor = np.random.randn(256).astype(np.float32)
    codebook = ref.build_nf4_codebook()

    want_q, want_abs = ref.quantize_blockwise(tensor, codebook, 64)

    try:
        from nf4.quantize import quantize_blockwise, dequantize_blockwise
        got_q, got_abs = quantize_blockwise(tensor, codebook, 64)
        if got_q.shape == want_q.shape and np.allclose(want_abs, got_abs, atol=1e-5):
            matches = np.sum(got_q == want_q)
            out["byte_exact_fraction"] = float(matches) / float(want_q.size)
        else:
            out["_note"] = "Quantization output dimensions or absmax mismatch."
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"quantize_blockwise failed: {e}"

    try:
        want_deq = ref.dequantize_blockwise(want_q, want_abs, codebook, 64)
        got_deq = dequantize_blockwise(want_q, want_abs, codebook, 64)
        if np.allclose(want_deq, got_deq, atol=1e-5):
            out["dequantize_match"] = 1.0
        else:
            if "_note" not in out:
                out["_note"] = "Dequantization output does not match reference."
    except Exception as e:
        pass

    try:
        from nf4.compare import compare_distributions
        res = compare_distributions()
        if isinstance(res, dict) and "normal" in res and "nf4" in res["normal"]:
            out["compare_runs"] = 1.0
    except Exception:
        pass

    return out
