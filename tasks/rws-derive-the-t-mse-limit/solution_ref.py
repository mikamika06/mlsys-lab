import numpy as np


def t_mse_limit(z_teacher, z_student, temperatures):
    z_teacher = np.asarray(z_teacher, dtype=np.float64)
    z_student = np.asarray(z_student, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)

    def softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    scaled_kl = []
    for t in temperatures:
        p_t = softmax(z_teacher / t)
        p_s = softmax(z_student / t)
        scaled_kl.append(t * t * np.sum(p_t * (np.log(p_t) - np.log(p_s))))

    zt = z_teacher - np.mean(z_teacher)
    zs = z_student - np.mean(z_student)
    limit = 0.5 * np.sum((zt - zs) ** 2)

    return np.asarray(scaled_kl, dtype=np.float64), np.float64(limit)
