import ref
from qlora.derive import expected_bits


def check(workdir):
    out = {"bits_matched": 0.0}
    b_bits = 4.0
    l_bits = 16.0
    b_params = 1000000
    l_params = 10000
    dq = True
    got = expected_bits(b_bits, l_bits, b_params, l_params, dq)
    want = ref.expected_bits(b_bits, l_bits, b_params, l_params, dq)
    if abs(got - want) < 1e-5:
        out["bits_matched"] = 1.0
    else:
        out["_note"] = f"expected {want}, got {got}"
    return out
