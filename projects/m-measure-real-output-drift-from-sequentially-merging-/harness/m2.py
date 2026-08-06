import ref
import numpy as np

def check(workdir):
    from adaptermerge.drift import compute_relative_error
    rng = np.random.default_rng(999)
    ok = 0
    total = 5
    for _ in range(total):
        ref_out = rng.standard_normal((16, 16))
        merged_out = ref_out + 0.01 * rng.standard_normal((16, 16))
        want = ref.compute_relative_error_ref(ref_out, merged_out)
        try:
            got = compute_relative_error(ref_out, merged_out)
            if got is not None and np.isclose(got, want, rtol=1e-5, atol=1e-5):
                ok += 1
        except Exception:
            pass
    matched = 1.0 if ok == total else 0.0
    return {"rel_err_matched": matched}
