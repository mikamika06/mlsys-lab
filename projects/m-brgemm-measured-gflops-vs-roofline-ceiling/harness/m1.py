import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from brgemm.dispatch import reconstruct_call_sequence

    total = float(len(ref.CONFIGS))
    mismatches = 0.0

    for cfg in ref.CONFIGS:
        want = ref.reconstruct_call_sequence(
            cfg["M"], cfg["N"], cfg["K"], cfg["mb"], cfg["nb"], cfg["kb"]
        )
        got = reconstruct_call_sequence(
            cfg["M"], cfg["N"], cfg["K"], cfg["mb"], cfg["nb"], cfg["kb"]
        )
        if got != want:
            mismatches += 1.0

    rel_err = mismatches / total
    out = {"rel_err": float(rel_err)}
    if mismatches > 0:
        out["_note"] = f"{int(mismatches)} of {int(total)} call sequences differed from reference"
    return out
