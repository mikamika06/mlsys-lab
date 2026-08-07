import ref
import numpy as np


def check(workdir):
    from bnb_quant.outliers import identify_outliers

    tensor = ref.generate_test_tensor()
    want = ref.reference_outliers(tensor, threshold=5.0)
    try:
        got = identify_outliers(tensor, threshold=5.0)
    except Exception as e:
        return {"outliers_identified": 0.0, "_note": str(e)}

    if got is None:
        return {"outliers_identified": 0.0, "_note": "returned None"}

    got_arr = np.asarray(got, dtype=bool)
    if np.array_equal(got_arr, want):
        return {"outliers_identified": 1.0}
    return {"outliers_identified": 0.0, "_note": "outlier mask mismatch"}
