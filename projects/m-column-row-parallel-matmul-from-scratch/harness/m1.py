import sys
import os
import ref
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    from tpp.parallel import column_parallel_matmul, row_parallel_matmul

    cases = ref.get_test_cases()
    matched = 0.0
    for x, w, bias in cases:
        want_col = ref.reference_column_parallel(x, w, bias)
        got_col = column_parallel_matmul(x, w, bias)

        want_row = ref.reference_row_parallel(x, w, bias)
        got_row = row_parallel_matmul(x, w, bias)

        if got_col is not None and np.max(np.abs(got_col - want_col)) < 1e-5:
            matched += 0.5
        if got_row is not None and np.max(np.abs(got_row - want_row)) < 1e-5:
            matched += 0.5

    return {"matmul_matched": float(matched)}
