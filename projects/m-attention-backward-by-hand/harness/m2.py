import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"flops_match": 0.0, "overhead_match": 0.0}
    try:
        from attnbwd.overhead import compute_attention_flops, recompute_overhead
    except Exception as e:
        out["_note"] = f"Failed to import from attnbwd.overhead: {e}"
        return out

    cases = [
        (1, 4, 128, 64),
        (2, 8, 256, 128),
        (4, 16, 512, 64),
    ]

    flops_ok = True
    overhead_ok = True

    try:
        for B, H, N, D in cases:
            for pass_type in ["forward", "backward"]:
                for recomp in [True, False]:
                    if pass_type == "forward" and not recomp:
                        continue
                    want = ref.compute_attention_flops(B, H, N, D, pass_type, recomp)
                    got = compute_attention_flops(B, H, N, D, pass_type, recomp)
                    if got != want:
                        flops_ok = False
                        if "_note" not in out:
                            out["_note"] = f"FLOP mismatch: got {got}, want {want} for ({B},{H},{N},{D},{pass_type},{recomp})"

            want_oh = ref.recompute_overhead(B, H, N, D)
            got_oh = recompute_overhead(B, H, N, D)
            for k, v in want_oh.items():
                if k not in got_oh or abs(got_oh[k] - v) > 1e-6:
                    overhead_ok = False
                    if "_note" not in out:
                        out["_note"] = f"Overhead mismatch for key {k}: got {got_oh.get(k)}, want {v}"
    except Exception as e:
        out["_note"] = f"Execution error: {e}"
        return out

    out["flops_match"] = 1.0 if flops_ok else 0.0
    out["overhead_match"] = 1.0 if overhead_ok else 0.0
    return out
