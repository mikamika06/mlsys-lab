import sys
from pathlib import Path

task_dir = str(Path(__file__).parent.resolve())
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

import solution_ref


def grade(sol, fx) -> dict:
    test_cases = [
        ([16, 32, 64], 128, 64),
        ([8, 16, 32], 64, 32),
    ]

    max_abs_err = 0.0

    for block_sizes, seq_len, d_model in test_cases:
        try:
            ref_out = solution_ref.attention_divergence(
                block_sizes, seq_len=seq_len, d_model=d_model
            )
            student_out = sol.attention_divergence(
                block_sizes, seq_len=seq_len, d_model=d_model
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if not isinstance(student_out, list) or len(student_out) != len(ref_out):
            return {"max_abs_err": float("inf")}

        for s_val, r_val in zip(student_out, ref_out):
            err = abs(s_val - r_val)
            if err > max_abs_err:
                max_abs_err = err

    return {"max_abs_err": max_abs_err}
