import ref
import numpy as np

def check(workdir):
    from iqquant.superblocks import decode_iq4_xs
    from iqquant.bpw import compute_bpw

    block = bytes([50] + list(range(32)))
    want_dec = ref.ref_decode_iq4_xs(block)
    got_dec = decode_iq4_xs(block)
    dequant_match = 1.0 if np.allclose(want_dec, got_dec, atol=1e-5) else 0.0

    bpw_match = 1.0
    for t in ["IQ1_S", "IQ2_XXS", "IQ4_XS", "TQ1_0", "TQ2_0"]:
        if compute_bpw(t) != ref.ref_compute_bpw(t):
            bpw_match = 0.0
            break

    out = {"dequant_match": dequant_match, "bpw_match": bpw_match}
    return out
