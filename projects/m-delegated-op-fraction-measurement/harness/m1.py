import ref
import numpy as np

def check(workdir):
    try:
        from delegate_measure.extractor import measure_delegation
    except ImportError:
        return {"rel_err": 1.0}

    errs = []
    for ops in ref.MODELS:
        part_ops = ref.partition_xnnpack(ops)
        want = ref.measure_delegation(part_ops)
        try:
            got = measure_delegation(part_ops)
            err = abs(want - got) / (abs(want) + 1e-9)
            errs.append(err)
        except Exception:
            errs.append(1.0)

    return {"rel_err": float(np.mean(errs)) if errs else 1.0}
