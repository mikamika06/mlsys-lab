import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from quantplan.backend import will_fallback_to_cpu

    out = {"fallback_matches": 0.0}
    total = len(ref.FALLBACK_SCENARIOS)
    matched = 0
    for i, (qtype, bcfg, want) in enumerate(ref.FALLBACK_SCENARIOS):
        got = will_fallback_to_cpu(qtype, bcfg)
        if got == want:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i} ({qtype}, {bcfg['name']}): got {got}, expected {want}"

    if matched == total:
        out["fallback_matches"] = 1.0
    return out
