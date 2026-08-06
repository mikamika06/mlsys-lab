import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ref
import reference.gguf_utils.perf as ref_perf


def check(workdir):
    from gguf_utils.perf import fit_decode_performance

    fixtures = ref.get_perf_fixtures()
    out = {"fit_match": 0.0}
    for data in fixtures:
        want = ref_perf.fit_decode_performance(data)
        try:
            got = fit_decode_performance(data)
        except Exception:
            got = {}
        if isinstance(got, dict) and "slope" in got and "intercept" in got:
            if abs(got["slope"] - want["slope"]) < 1e-5 and abs(got["intercept"] - want["intercept"]) < 1e-5:
                out["fit_match"] = 1.0
            else:
                out["_note"] = f"got slope {got.get('slope')}, want {want['slope']}"
        else:
            out["_note"] = "fit_decode_performance did not return valid slope/intercept dict"
    return out
