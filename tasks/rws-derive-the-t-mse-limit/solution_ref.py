import math
import numpy as np


def t_mse_limit(z_teacher, z_student, temperatures):
    z_teacher = np.asarray(z_teacher, dtype=np.float64)
    z_student = np.asarray(z_student, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)

    def softmax(x, t):
        n = len(x)
        max_val = x[0] / t
        for i in range(1, n):
            val = x[i] / t
            if val > max_val:
                max_val = val

        e = []
        sum_e = 0.0
        for i in range(n):
            val = math.exp((x[i] / t) - max_val)
            e.append(val)
            sum_e += val

        return [val / sum_e for val in e]

    scaled_kl = []
    for t in temperatures:
        p_t = softmax(z_teacher, t)
        p_s = softmax(z_student, t)

        kl_sum = 0.0
        for i in range(len(p_t)):
            pt_i = p_t[i]
            ps_i = p_s[i]
            kl_sum += pt_i * (math.log(pt_i) - math.log(ps_i))

        scaled_kl.append(t * t * kl_sum)

    n = len(z_teacher)
    sum_zt = 0.0
    for i in range(n):
        sum_zt += z_teacher[i]
    mean_zt = sum_zt / n

    sum_zs = 0.0
    for i in range(n):
        sum_zs += z_student[i]
    mean_zs = sum_zs / n

    limit_sum = 0.0
    for i in range(n):
        zt_i = z_teacher[i] - mean_zt
        zs_i = z_student[i] - mean_zs
        diff = zt_i - zs_i
        limit_sum += diff * diff

    limit = 0.5 * limit_sum

    return np.asarray(scaled_kl, dtype=np.float64), np.float64(limit)
