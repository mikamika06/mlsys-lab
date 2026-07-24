"""Deterministic teacher distribution and a perturbation sweep of student
distributions (shared vocabulary, same index alignment) for comparing ULD
(sorted-Wasserstein-1) and standard same-vocab KL. Row 0 of the sweep is
exactly the teacher distribution (the alpha == 0 / exact-match case).

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

V = 6
ALPHAS = [0.0, 0.02, -0.02, 0.05, -0.05, 0.08, -0.08, 0.1, -0.1]


def main() -> None:
    rng = np.random.default_rng(0)

    raw = rng.uniform(0.5, 3.0, size=V)
    p_teacher = raw / raw.sum()

    # a zero-sum perturbation direction, small enough to keep every swept
    # student distribution strictly positive across the whole alpha range
    d = rng.standard_normal(V)
    d = d - d.mean()
    d = d / np.max(np.abs(d)) * (p_teacher.min() * 0.6)

    students = []
    for a in ALPHAS:
        row = p_teacher + a * d
        assert np.all(row > 0), "perturbation left the simplex interior"
        assert abs(row.sum() - 1.0) < 1e-12
        students.append(row)
    P_students = np.stack(students, axis=0)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "uld_p_teacher.npy", p_teacher)
    np.save(out / "uld_p_students.npy", P_students)


if __name__ == "__main__":
    main()
