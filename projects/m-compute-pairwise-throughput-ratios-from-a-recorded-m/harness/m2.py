import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from benchcomp.scaling import compute_scaling_efficiency

    records = ref.BENCHMARK_RECORDS
    want_eff = ref.compute_scaling_efficiency(records)
    got_eff = compute_scaling_efficiency(records)

    out = {"scaling_rel_err": 1.0}

    errs = []
    for key, want_v in want_eff.items():
        if key not in got_eff:
            errs.append(1.0)
        else:
            got_v = got_eff[key]
            errs.append(abs(got_v - want_v) / (abs(want_v) + 1e-12))

    max_err = max(errs) if errs else 1.0
    out["scaling_rel_err"] = float(max_err)

    return out
