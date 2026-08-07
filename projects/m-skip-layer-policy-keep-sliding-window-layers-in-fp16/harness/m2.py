import os
import sys
import numpy as np

def check(workdir):
    sys.path.insert(0, os.path.dirname(__file__))
    import ref
    sys.path.pop(0)

    out = {"sim_match": 0.0, "err_match": 0.0, "err_lower_than_fp8": 0.0}
    sys.path.insert(0, workdir)
    try:
        from policy.eval import simulate_kv_cache_output, eval_rel_err

        cfg = [
            {"index": 0, "kind": "full"},
            {"index": 1, "kind": "sliding", "window": 128},
            {"index": 2, "kind": "full"},
            {"index": 3, "kind": "sliding", "window": 128}
        ]
        dtypes = ref.assign_kv_dtypes(cfg)

        want_sim = ref.simulate_kv_cache_output(cfg, dtypes, 256, 64)
        got_sim = simulate_kv_cache_output(cfg, dtypes, 256, 64)
        if np.allclose(want_sim, got_sim, rtol=1e-3, atol=1e-3):
            out["sim_match"] = 1.0
        else:
            out["_note"] = "simulate_kv_cache_output outputs do not match reference"
            return out

        want_err = ref.eval_rel_err(cfg, dtypes, 256, 64)
        got_err = eval_rel_err(cfg, dtypes, 256, 64)

        if abs(want_err - got_err) < 1e-4:
            out["err_match"] = 1.0
        else:
            out["_note"] = f"err mismatch: got {got_err}, want {want_err}"
            return out

        all_fp8 = ["float8"] * len(cfg)
        got_err_fp8 = eval_rel_err(cfg, all_fp8, 256, 64)
        if got_err < got_err_fp8:
            out["err_lower_than_fp8"] = 1.0
        else:
            out["_note"] = f"skip-layer err {got_err} not strictly less than all-fp8 err {got_err_fp8}"

    except Exception as e:
        out["_note"] = f"error: {type(e).__name__} - {str(e)}"
    finally:
        sys.path.pop(0)

    return out
