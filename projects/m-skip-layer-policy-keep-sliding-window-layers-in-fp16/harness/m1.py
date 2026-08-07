import os
import sys

def check(workdir):
    sys.path.insert(0, os.path.dirname(__file__))
    import ref
    sys.path.pop(0)

    out = {"policy_match": 0.0, "bytes_match": 0.0}
    sys.path.insert(0, workdir)
    try:
        from policy.quant import assign_kv_dtypes, compute_kv_bytes

        cfg = [
            {"index": 0, "kind": "full"},
            {"index": 1, "kind": "sliding", "window": 2048},
            {"index": 2, "kind": "full"}
        ]

        want_dtypes = ref.assign_kv_dtypes(cfg)
        got_dtypes = assign_kv_dtypes(cfg)
        if got_dtypes == want_dtypes:
            out["policy_match"] = 1.0
        else:
            out["_note"] = f"policy mismatch: got {got_dtypes}, want {want_dtypes}"
            return out

        want_b = ref.compute_kv_bytes(cfg, want_dtypes, 4096, 2, 8, 128)
        got_b = compute_kv_bytes(cfg, want_dtypes, 4096, 2, 8, 128)
        if want_b == got_b:
            out["bytes_match"] = 1.0
        else:
            out["_note"] = f"bytes mismatch: got {got_b}, want {want_b}"

    except Exception as e:
        out["_note"] = f"error: {type(e).__name__} - {str(e)}"
    finally:
        sys.path.pop(0)

    return out
