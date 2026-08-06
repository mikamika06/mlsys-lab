import ref
import numpy as np

def check(workdir):
    from adaptermerge.merge import merge_adapters
    cases = ref.get_test_cases()
    ok = 0
    for i, c in enumerate(cases):
        want = ref.merge_adapters_ref(c["w_base"], c["delta1"], c["delta2"], c["scale1"], c["scale2"])
        try:
            got = merge_adapters(c["w_base"], c["delta1"], c["delta2"], c["scale1"], c["scale2"])
            if got is not None and np.allclose(got, want, rtol=1e-5, atol=1e-5):
                ok += 1
        except Exception:
            pass
    matched = 1.0 if ok == len(cases) else 0.0
    return {"weights_matched": matched}
