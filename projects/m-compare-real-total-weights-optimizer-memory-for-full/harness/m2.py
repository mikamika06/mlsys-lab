import os
import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"memory_match": 0.0}
    try:
        from optmem import states
        configs = ref.get_configs()
        ok = 0
        for i, cfg in enumerate(configs):
            for mode in ("full", "lora"):
                want = ref.compute_optimizer_bytes(cfg, mode)
                got = states.compute_optimizer_bytes(cfg, mode)
                if got == want:
                    ok += 1
                elif "_note" not in out:
                    out["_note"] = f"config {i} mode {mode}: got {got}, reference {want}"
        total_checks = len(configs) * 2
        out["memory_match"] = 1.0 if ok == total_checks else 0.0
    except Exception as e:
        out["_note"] = f"Error during check: {type(e).__name__}: {e}"
    return out
