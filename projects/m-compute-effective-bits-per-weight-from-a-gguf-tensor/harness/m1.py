import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ref
import reference.gguf_utils.bits as ref_bits


def check(workdir):
    from gguf_utils.bits import compute_effective_bits

    fixtures = ref.get_tensor_fixtures()
    out = {"bits_matched": 0.0, "configs": float(len(fixtures))}
    ok = 0
    for i, tensors in enumerate(fixtures):
        want = ref_bits.compute_effective_bits(tensors)
        try:
            got = compute_effective_bits(tensors)
        except Exception:
            got = -1.0
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"fixture {i}: got {got}, reference {want}"
    out["bits_matched"] = float(ok)
    return out
