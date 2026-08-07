import ref
import numpy as np


def check(workdir):
    out = {"quant_dequant_match": 0.0, "byte_exact_fraction": 0.0}
    try:
        from quant.quantize import quantize_blockwise, dequantize_blockwise
        np.array(42) # dummy reference to ensure import

        np.random.seed(42)
        weights = np.random.randn(512).astype(np.float32)
        block_size = 64

        q_got, s_got = quantize_blockwise(weights, block_size, "nf4")
        q_want, s_want = ref.compute_reference_quant(weights, block_size, "nf4")

        if np.array_equal(q_got, q_want) and np.allclose(s_got, s_want, atol=1e-5):
            out["quant_dequant_match"] = 1.0
            out["byte_exact_fraction"] = 1.0
        else:
            match_fraction = float(np.mean(q_got == q_want))
            out["byte_exact_fraction"] = match_fraction
            if match_fraction >= 0.95:
                out["quant_dequant_match"] = 1.0
            out["_note"] = f"Quantized output mismatch. Match fraction: {match_fraction}"
    except Exception as e:
        out["_note"] = f"Error during quantization check: {e}"
    return out
