import ref
import numpy as np

def check(workdir):
    m = {"perplexity_delta_ok": 0.0}
    try:
        from int4.eval import compute_perplexity
        w = ref.get_test_weights((64, 64))
        inputs = np.random.randn(10, 64).astype(np.float32)
        base_ppl = compute_perplexity(w, inputs)

        from int4.quant import quantize_weights
        packed, scale, shape = quantize_weights(w, group_size=64)
        even = packed & 0x0F
        odd = (packed >> 4) & 0x0F
        unpacked = np.empty(packed.size * 2, dtype=np.uint8)
        unpacked[0::2] = even
        unpacked[1::2] = odd
        unpacked = unpacked[:np.prod(shape)]
        q = unpacked.astype(np.int8) - 8
        w_q = (q.reshape(-1, 64).astype(np.float32) * scale).reshape(shape)

        quant_ppl = compute_perplexity(w_q, inputs)
        if abs(quant_ppl - base_ppl) / (base_ppl + 1e-5) < 0.5:
            m["perplexity_delta_ok"] = 1.0
    except Exception:
        pass
    return m
