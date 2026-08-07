import ref
import numpy as np

def check(workdir):
    from bnb_sim.quant import quantize_int8
    cases = ref.generate_cases()
    matched = 0
    for case in cases:
        want_q, want_s, want_o = ref.quantize_int8(case)
        got_q, got_s, got_o = quantize_int8(case)
        if (got_q.shape == want_q.shape and
            np.allclose(got_s, want_s, atol=1e-3) and
            np.allclose(got_o, want_o, atol=1e-3) and
            np.array_equal(got_q, want_q)):
            matched += 1
    out = {"quant_matched": float(matched)}
    return out
