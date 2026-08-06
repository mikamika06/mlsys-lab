import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from longrope.scaling import compute_longrope_factors

    out = {"factors_matched": 0.0, "rel_err": 1.0}
    configs = ref.get_test_configs()

    max_err = 0.0
    matched = 0

    for cfg in configs:
        want = ref.oracle_longrope_factors(
            cfg["head_dim"], cfg["orig_len"], cfg["target_len"], cfg["base"]
        )
        got = compute_longrope_factors(
            cfg["head_dim"], cfg["orig_len"], cfg["target_len"], cfg["base"]
        )

        err = np.max(np.abs(got - want) / (np.abs(want) + 1e-12))
        max_err = max(max_err, float(err))

        if err <= 1e-5:
            matched += 1

    out["rel_err"] = float(max_err)
    if matched == len(configs):
        out["factors_matched"] = 1.0
    else:
        out["_note"] = f"Matched {matched}/{len(configs)} configs. Max rel err: {max_err:.6e}"

    return out
