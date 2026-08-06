import numpy as np
import ref


def check(workdir):
    from pagecache.estimator import estimate_resident_bytes

    out = {"rel_err": 0.0}
    total_err = 0.0
    count = 0

    for fixture in ref.FIXTURES_M1:
        want = ref.estimate_resident_bytes(
            fixture["file_size"], fixture["accesses"]
        )
        got = estimate_resident_bytes(
            fixture["file_size"], fixture["accesses"]
        )
        err = abs(got - want) / float(want) if want > 0 else 0.0
        total_err += err
        count += 1

    out["rel_err"] = float(total_err / count) if count > 0 else 0.0
    return out
