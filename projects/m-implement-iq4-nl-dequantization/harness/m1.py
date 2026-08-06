import ref
import numpy as np

def check(workdir):
    from iqquant.dequant import dequantize_iq4_nl
    data = bytes([0x12, 0xAB, 0xF0, 0x34])
    scales = np.array([1.5], dtype=np.float32)
    want = ref.ref_dequantize_iq4_nl(data, scales)
    got = dequantize_iq4_nl(data, scales)
    err = float(np.max(np.abs(want - got)))
    out = {"max_abs_err": err}
    if err > 1e-5:
        out["_note"] = f"max abs error {err} exceeds threshold"
    return out
