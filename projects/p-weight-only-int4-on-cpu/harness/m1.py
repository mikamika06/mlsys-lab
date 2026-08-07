import ref
import numpy as np

def check(workdir):
    m = {"size_ok": 0.0, "reconstruct_ok": 0.0}
    try:
        from int4.quant import quantize_weights
        w = ref.get_test_weights((128, 128))
        packed, scale, shape = quantize_weights(w, group_size=128)
        oracle_size = ref.get_oracle_size(w)
        if packed.nbytes + scale.nbytes <= oracle_size * 1.05:
            m["size_ok"] = 1.0

        even = packed & 0x0F
        odd = (packed >> 4) & 0x0F
        unpacked = np.empty(packed.size * 2, dtype=np.uint8)
        unpacked[0::2] = even
        unpacked[1::2] = odd
        unpacked = unpacked[:np.prod(shape)]
        q = unpacked.astype(np.int8) - 8
        dequant = q.reshape(-1, 128).astype(np.float32) * scale
        err = np.mean(np.abs(w - dequant))
        if err < 0.2:
            m["reconstruct_ok"] = 1.0
    except Exception:
        pass
    return m
