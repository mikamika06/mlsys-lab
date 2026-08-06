import ref
import sys
import os

def check(workdir):
    sys.path.insert(0, workdir)
    import specbatch.flops as flops
    import specbatch.crossover as crossover

    out = {"flops_matched": 0.0, "cross_matched": 0.0}

    flops_ok = 0
    for b, c, n in ref.FLOPS_CASES:
        want = ref.flops_neutral_batch_size(b, c, n)
        try:
            got = flops.flops_neutral_batch_size(b, c, n)
        except NotImplementedError:
            continue
        if want == got:
            flops_ok += 1
        else:
            if "_note" not in out:
                out["_note"] = f"flops_neutral({b}, {c}, {n}): got {got}, want {want}"
    out["flops_matched"] = float(flops_ok)

    cross_ok = 0
    for sweep in ref.SWEEPS:
        want = ref.find_crossover_batch_size(sweep)
        try:
            got = crossover.find_crossover_batch_size(sweep)
        except NotImplementedError:
            continue
        if want == got:
            cross_ok += 1
        else:
            if "_note" not in out:
                out["_note"] = f"sweep {sweep}: got {got}, want {want}"
    out["cross_matched"] = float(cross_ok)

    return out
