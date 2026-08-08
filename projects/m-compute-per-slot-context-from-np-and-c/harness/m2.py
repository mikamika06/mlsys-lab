import sys

import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from llamaslot.saturation import find_np_saturation

    out = {
        "saturations_matched": 0.0,
        "total_cases": float(len(ref.SATURATION_CONFIGS)),
    }
    ok = 0
    for cfg in ref.SATURATION_CONFIGS:
        want = ref.find_np_saturation(
            cfg["total_ctx"], cfg["req_slot_ctx"], cfg["gpu_slot_cap"]
        )
        got = find_np_saturation(
            cfg["total_ctx"], cfg["req_slot_ctx"], cfg["gpu_slot_cap"]
        )
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cfg {cfg}: got {got}, want {want}"
    out["saturations_matched"] = float(ok)
    return out
