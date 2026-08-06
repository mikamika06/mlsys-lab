import ref
import numpy as np


def check(workdir):
    from quant.round import weighted_round_to_nearest
    fixtures = ref.get_test_fixtures()
    ok = 0
    q_min, q_max = -8, 7
    for i, (w, im) in enumerate(fixtures):
        want_s, want_q = ref.weighted_round_to_nearest(w, im, q_min, q_max)
        try:
            got_s, got_q = weighted_round_to_nearest(w, im, q_min, q_max)
            if abs(float(got_s) - want_s) < 1e-3 and np.allclose(got_q, want_q, atol=1e-5):
                ok += 1
        except Exception:
            pass
    out = {"grid_matched": 1.0 if ok == len(fixtures) else 0.0}
    if ok < len(fixtures):
        out["_note"] = f"matched {ok}/{len(fixtures)} grid search fixtures"
    return out
