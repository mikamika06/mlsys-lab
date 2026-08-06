import ref
import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"q4_1_exact_matches": 0.0, "q5_1_exact_matches": 0.0}

    try:
        from qblocks.q4_1 import dequantize_q4_1, quantize_q4_1
        from qblocks.q5_1 import dequantize_q5_1, quantize_q5_1
    except Exception as e:
        out["_note"] = f"Failed to import qblocks modules: {e}"
        return out

    data = ref.generate_synthetic_data(seed=202, size=256)

    try:
        ref_b41 = ref.ref_quantize_q4_1(data)
        ref_dq41 = ref.ref_dequantize_q4_1(ref_b41)
        usr_b41 = quantize_q4_1(data)
        usr_dq41 = dequantize_q4_1(usr_b41)

        b41_ok = (
            len(ref_b41) == len(usr_b41)
            and all(
                abs(u["d"] - r["d"]) < 1e-4
                and abs(u["m"] - r["m"]) < 1e-4
                and np.array_equal(u["qs"], r["qs"])
                for u, r in zip(usr_b41, ref_b41)
            )
            and np.allclose(usr_dq41, ref_dq41, atol=1e-5)
        )
        if b41_ok:
            out["q4_1_exact_matches"] = 1.0
    except Exception as e:
        out["_note"] = f"Q4_1 check error: {e}"
        return out

    try:
        ref_b51 = ref.ref_quantize_q5_1(data)
        ref_dq51 = ref.ref_dequantize_q5_1(ref_b51)
        usr_b51 = quantize_q5_1(data)
        usr_dq51 = dequantize_q5_1(usr_b51)

        b51_ok = (
            len(ref_b51) == len(usr_b51)
            and all(
                abs(u["d"] - r["d"]) < 1e-4
                and abs(u["m"] - r["m"]) < 1e-4
                and int(u["qh"]) == int(r["qh"])
                and np.array_equal(u["qs"], r["qs"])
                for u, r in zip(usr_b51, ref_b51)
            )
            and np.allclose(usr_dq51, ref_dq51, atol=1e-5)
        )
        if b51_ok:
            out["q5_1_exact_matches"] = 1.0
    except Exception as e:
        out["_note"] = f"Q5_1 check error: {e}"
        return out

    return out
