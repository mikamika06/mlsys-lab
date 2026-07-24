import numpy as np


def uld_and_kl_along_sweep(p_teacher: np.ndarray, p_students: np.ndarray):
    p_teacher = np.asarray(p_teacher, dtype=np.float64)
    p_students = np.asarray(p_students, dtype=np.float64)

    t_sorted = np.sort(p_teacher)
    n = p_students.shape[0]
    uld = np.empty(n, dtype=np.float64)
    kl = np.empty(n, dtype=np.float64)

    for i in range(n):
        row = p_students[i]
        uld[i] = np.sum(np.abs(t_sorted - np.sort(row)))
        kl[i] = np.sum(p_teacher * np.log(p_teacher / row))

    return uld, kl
