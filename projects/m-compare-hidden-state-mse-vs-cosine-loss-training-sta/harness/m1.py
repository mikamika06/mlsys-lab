import ref
import numpy as np


def check(workdir):
    from distill.loss import mse_loss, cosine_loss
    s_states, t_states = ref.generate_fixtures()
    out = {"loss_match": 0.0}
    ok = 0

    cases = [
        (s_states, t_states),
        (s_states * 0.5, t_states * 2.0),
        (np.ones_like(s_states), np.zeros_like(t_states)),
        (s_states, s_states),
        (np.random.RandomState(1).randn(2, 8, 32), np.random.RandomState(2).randn(2, 8, 32))
    ]

    for i, (s, t) in enumerate(cases):
        want_mse = ref.ref_mse_loss(s, t)
        got_mse = float(mse_loss(s, t))
        want_cos = ref.ref_cosine_loss(s, t)
        got_cos = float(cosine_loss(s, t))

        if abs(want_mse - got_mse) < 1e-5 and abs(want_cos - got_cos) < 1e-5:
            ok += 2
        elif "_note" not in out:
            out["_note"] = f"case {i}: got mse={got_mse} (want {want_mse}), cos={got_cos} (want {want_cos})"

    out["loss_match"] = float(ok)
    return out
