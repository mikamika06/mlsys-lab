import ref
import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"q4_0_exact_matches": 0.0}
    try:
        from qblocks.q4_0 import dequantize_q4_0, quantize_q4_0
    except Exception as e:
        out["_note"] = f"Failed to import qblocks.q4_0: {e}"
        return out

    data = ref.generate_synthetic_data(seed=101, size=256)
    ref_blocks = ref.ref_quantize_q4_0(data)
    ref_dequant = ref.ref_dequantize_q4_0(ref_blocks)

    try:
        user_blocks = quantize_q4_0(data)
        user_dequant = dequantize_q4_0(user_blocks)
    except Exception as e:
        out["_note"] = f"Execution error in Q4_0 quant/dequant: {e}"
        return out

    if len(user_blocks) != len(ref_blocks):
        out["_note"] = f"Block count mismatch: got {len(user_blocks)}, expected {len(ref_blocks)}"
        return out

    blocks_match = True
    for ub, rb in zip(user_blocks, ref_blocks):
        if not (abs(ub["d"] - rb["d"]) < 1e-4 and np.array_equal(ub["qs"], rb["qs"])):
            blocks_match = False
            break

    dequant_match = bool(np.allclose(user_dequant, ref_dequant, atol=1e-5))

    if blocks_match and dequant_match:
        out["q4_0_exact_matches"] = 1.0
    else:
        out["_note"] = "Q4_0 output did not match reference quantization structure or dequantized output"

    return out
