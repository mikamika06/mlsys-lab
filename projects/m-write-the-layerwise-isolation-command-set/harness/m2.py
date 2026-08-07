import sys

import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from polyiso.divergence import find_first_divergent_layer
    from polyiso.stats import compute_polygraphy_stats

    out = {"divergence_matched": 0.0, "stats_matched": 0.0}

    div_ok = 0
    for tc in ref.TEST_CASES_DIVERGENCE:
        want = ref.find_first_divergent_layer(tc["layers"], rtol=tc["rtol"], atol=tc["atol"])
        got = find_first_divergent_layer(tc["layers"], rtol=tc["rtol"], atol=tc["atol"])
        if got == want:
            div_ok += 1
        elif "_note" not in out:
            out["_note"] = f"divergence mismatch: got {got}, want {want}"

    if div_ok == len(ref.TEST_CASES_DIVERGENCE):
        out["divergence_matched"] = 1.0

    stats_ok = 0
    for tc in ref.TEST_CASES_STATS:
        want_s = ref.compute_polygraphy_stats(tc["a"], tc["b"])
        got_s = compute_polygraphy_stats(tc["a"], tc["b"])
        if isinstance(got_s, dict) and all(
            k in got_s and np.isclose(got_s[k], want_s[k], rtol=1e-5) for k in want_s
        ):
            stats_ok += 1
        elif "_note" not in out:
            out["_note"] = f"stats mismatch: got {got_s}, want {want_s}"

    if stats_ok == len(ref.TEST_CASES_STATS):
        out["stats_matched"] = 1.0

    return out
