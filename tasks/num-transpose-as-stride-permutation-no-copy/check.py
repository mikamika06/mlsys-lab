import numpy as np
from mlsys import scorers


def grade(sol, fx) -> dict:
    cases = [
        (np.arange(24, dtype=np.int64).reshape(4, 6), (1, 0)),
        (np.arange(60, dtype=np.float64).reshape(3, 4, 5), (2, 0, 1)),
        (np.arange(48, dtype=np.int32).reshape(2, 3, 8), (1, 2, 0)),
        (np.arange(18, dtype=np.float64).reshape(3, 2, 3), (2, 1, 0)),
    ]

    score = 1.0
    for A, axes in cases:
        try:
            got = sol.transpose_view(A, axes)
            ref = np.transpose(A, axes=axes)

            if not np.shares_memory(A, got):
                score = 0.0
                break

            if got.shape != ref.shape or got.strides != ref.strides:
                score = 0.0
                break

            materialized_got = np.asarray(got).copy()
            materialized_ref = np.asarray(ref).copy()

            score *= scorers.byte_exact_fraction(
                materialized_ref,
                materialized_got,
            )
        except Exception:
            score = 0.0
            break

    return {"byte_exact_fraction": score}
