import ref
import time
import numpy as np

def check(workdir):
    m = {"decode_speedup_ok": 0.0}
    try:
        from int4.quant import quantize_weights
        w = ref.get_test_weights((256, 256))
        packed, scale, shape = quantize_weights(w)

        start = time.time()
        for _ in range(100):
            even = packed & 0x0F
            odd = (packed >> 4) & 0x0F
            unpacked = np.empty(packed.size * 2, dtype=np.uint8)
            unpacked[0::2] = even
            unpacked[1::2] = odd
            unpacked = unpacked[:np.prod(shape)]
            q = unpacked.astype(np.int8) - 8
            _ = q.reshape(-1, 128).astype(np.float32) * scale
        duration = time.time() - start
        if duration < 1.0:
            m["decode_speedup_ok"] = 1.0
    except Exception:
        pass
    return m
