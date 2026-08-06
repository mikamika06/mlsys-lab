import math


def softmax_vjp(x: list[float], g: list[float]) -> list[float]:
    max_x = x[0]
    for val in x:
        if val > max_x:
            max_x = val

    e_list = []
    for val in x:
        e_list.append(math.exp(val - max_x))

    sum_e = 0.0
    for val in e_list:
        sum_e += val

    s_list = []
    for val in e_list:
        s_list.append(val / sum_e)

    sum_gs = 0.0
    for g_val, s_val in zip(g, s_list):
        sum_gs += g_val * s_val

    res_list = []
    for s_val, g_val in zip(s_list, g):
        res_list.append(s_val * (g_val - sum_gs))

    return res_list
