import ref
import numpy as np


def check(workdir):
    from quant.scale import optimal_scale
    fixtures = ref.get_test_fixtures()
    ok = 0
    q_min, q_max = -8, 7
    for i, (w, im) in enumerate(fixtures):
        want = ref.optimal_scale(w, im, q_min, q_max)
        try:
            got = optimal_scale(w, im, q_min, q_max)
            if abs(float(got) - want) < 1e-4:
                ok += 1
        except Exception:
            pass
    out = {"scale_matched": 1.0 if ok == len(fixtures) else 0.0}
    if ok < len(fixtures):
        out["_note"] = f"matched {ok}/{len(fixtures)} scale fixtures"
    return out
