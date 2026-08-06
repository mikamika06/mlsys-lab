import math
import numpy as np


def softmax_vjp(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    
    max_x = x.flat[0]
    for val in x.flat:
        if val > max_x:
            max_x = val
            
    e_list = []
    for val in x.flat:
        e_list.append(math.exp(val - max_x))
        
    sum_e = 0.0
    for val in e_list:
        sum_e += val
        
    s_list = []
    for val in e_list:
        s_list.append(val / sum_e)
        
    sum_gs = 0.0
    for g_val, s_val in zip(g.flat, s_list):
        sum_gs += g_val * s_val
        
    res_list = []
    for s_val, g_val in zip(s_list, g.flat):
        res_list.append(s_val * (g_val - sum_gs))
        
    return np.array(res_list, dtype=np.float64).reshape(x.shape)
