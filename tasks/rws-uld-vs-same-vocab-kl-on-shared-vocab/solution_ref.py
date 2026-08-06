import math
import numpy as np


def uld_and_kl_along_sweep(p_teacher: np.ndarray, p_students: np.ndarray):
    p_teacher = np.asarray(p_teacher, dtype=np.float64)
    p_students = np.asarray(p_students, dtype=np.float64)

    t_list = sorted(p_teacher.tolist())
    n = p_students.shape[0]
    uld = np.empty(n, dtype=np.float64)
    kl = np.empty(n, dtype=np.float64)

    for i in range(n):
        row = p_students[i]
        r_list = sorted(row.tolist())

        uld_val = 0.0
        for t_val, r_val in zip(t_list, r_list):
            uld_val += abs(t_val - r_val)
        uld[i] = uld_val

        kl_val = 0.0
        d = p_teacher.shape[0]
        for j in range(d):
            pt = p_teacher[j]
            pr = row[j]
            kl_val += pt * math.log(pt / pr)
        kl[i] = kl_val

    return uld, kl
