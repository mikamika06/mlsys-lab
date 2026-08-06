import numpy as np
import ref


def check(workdir):
    from bf16num.bf16 import bf16_bits_to_fp32, fp32_to_bf16_bits, round_fp32_to_bf16

    data = ref.make_m1_dataset()
    want_bits = ref.fp32_to_bf16_bits(data)
    want_f32 = ref.round_fp32_to_bf16(data)

    try:
        got_bits = fp32_to_bf16_bits(data)
        got_f32 = round_fp32_to_bf16(data)
        got_back = bf16_bits_to_fp32(got_bits)
    except Exception as e:
        return {"byte_exact_fraction": 0.0, "_note": f"Execution error: {e}"}

    total = float(len(data))
    bits_match = (got_bits == want_bits)
    f32_match = (np.isnan(want_f32) & np.isnan(got_f32)) | (want_f32 == got_f32)
    back_match = (np.isnan(want_f32) & np.isnan(got_back)) | (want_f32 == got_back)

    exact = bits_match & f32_match & back_match
    match_count = float(np.sum(exact))
    frac = match_count / total

    out = {"byte_exact_fraction": frac}
    if frac < 1.0:
        out["_note"] = f"Matched {match_count}/{total} elements"
    return out
