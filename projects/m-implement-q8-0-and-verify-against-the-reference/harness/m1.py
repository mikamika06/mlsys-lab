import ref
import numpy as np

def check(workdir):
    from quant.legacy import quantize_q8_0, dequantize_q8_0
    weights = ref.generate_test_weights()
    try:
        encoded = quantize_q8_0(weights)
        decoded = dequantize_q8_0(encoded, weights.shape)
        if decoded.shape != weights.shape:
            return {"q8_0_match": 0.0, "_note": "shape mismatch"}
        diff = np.max(np.abs(weights - decoded))
        if diff < 0.2:
            return {"q8_0_match": 1.0}
        return {"q8_0_match": 0.0, "_note": f"max diff {diff} too high"}
    except Exception as e:
        return {"q8_0_match": 0.0, "_note": str(e)}
