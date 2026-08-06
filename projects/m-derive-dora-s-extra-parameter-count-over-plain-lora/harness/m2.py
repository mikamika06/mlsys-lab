import numpy as np
import ref


def check(workdir):
    from adapter.dora import dora_forward

    cases = ref.get_test_cases()
    matched = 0
    for c in cases:
        want = ref_dora_forward(c["w"], c["a"], c["b"], c["g"], c["alpha"], c["x"])
        got = dora_forward(c["w"], c["a"], c["b"], c["g"], c["alpha"], c["x"])
        if np.allclose(got, want, atol=1e-5, rtol=1e-5):
            matched += 1
    ok = 1.0 if matched == len(cases) else 0.0
    out = {"forward_matched": ok}
    return out


def ref_dora_forward(w, a, b, g, alpha, x):
    r = a.shape[0]
    delta = (b @ a) * (alpha / r)
    w_tot = w + delta
    norms = np.linalg.norm(w_tot, axis=1, keepdims=True)
    norm_w = w_tot / (norms + 1e-12)
    final_w = g[:, None] * norm_w
    return x @ final_w.T
