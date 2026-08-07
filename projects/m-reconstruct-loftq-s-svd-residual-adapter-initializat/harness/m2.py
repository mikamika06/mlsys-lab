import ref
import numpy as np


def check(workdir):
    from loftq.dora import dora_forward

    out = {"forward_matched": 0.0}
    ok = 0
    for case in ref.CASES:
        W = case["W"]
        r = case["rank"]
        x = case["x"]
        g = case["g"]
        W_q, A, B = ref.loftq_init(W, r)

        for use_dora in [False, True]:
            want = ref.dora_forward(x, W_q, A, B, g, use_dora)
            got = dora_forward(x, W_q, A, B, g, use_dora)
            if np.allclose(want, got, atol=1e-5):
                ok += 1
    total_checks = len(ref.CASES) * 2
    if ok == total_checks:
        out["forward_matched"] = 1.0
    else:
        out["_note"] = f"Forward matched {ok}/{total_checks} configurations"
    return out
