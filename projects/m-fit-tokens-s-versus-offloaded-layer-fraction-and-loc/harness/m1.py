import sys

import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from offload.profiler import find_offload_cliff

    out = {"cliffs_matched": 0.0}
    correct = 0
    total = len(ref.PROFILES)

    for i, profile in enumerate(ref.PROFILES):
        want = ref.find_offload_cliff(profile)
        got = find_offload_cliff(profile)
        if abs(got - want) < 1e-3:
            correct += 1
        elif "_note" not in out:
            out["_note"] = f"Profile {i}: want cliff at {want}, got {got}"

    if correct == total:
        out["cliffs_matched"] = 1.0
    return out
