import math


def uld_and_kl_along_sweep(p_teacher: list[float], p_students: list[list[float]]):
    t_list = sorted(p_teacher)
    n = len(p_students)
    uld = []
    kl = []

    for i in range(n):
        row = p_students[i]
        r_list = sorted(row)

        uld_val = 0.0
        for t_val, r_val in zip(t_list, r_list):
            uld_val += abs(t_val - r_val)
        uld.append(uld_val)

        kl_val = 0.0
        d = len(p_teacher)
        for j in range(d):
            pt = p_teacher[j]
            pr = row[j]
            kl_val += pt * math.log(pt / pr)
        kl.append(kl_val)

    return uld, kl
