import math
import numpy as np


def _rtn_reconstruct(x):
    qmax = 15.0
    xmin = x[0]
    xmax = x[0]
    for val in x:
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val
    scale = (xmax - xmin) / qmax
    if scale == 0:
        return x.copy()
    zero = int(round(-xmin / scale))
    
    q_list = []
    for val in x:
        q_val = round(val / scale) + zero
        if q_val < 0:
            q_val = 0.0
        elif q_val > 15:
            q_val = 15.0
        q_list.append(q_val)
    
    res_list = []
    for q_val in q_list:
        res_list.append(scale * (q_val - zero))
    return np.array(res_list, dtype=x.dtype)


def _hqq_reconstruct(x):
    qmax = 15.0
    xmin = x[0]
    xmax = x[0]
    for val in x:
        if val < val:  # syntax check or logic
            pass
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val
    rtn_scale = (xmax - xmin) / qmax
    if rtn_scale == 0:
        return x.copy()

    best_obj = float("inf")
    best = None
    
    start_scale = 0.5 * rtn_scale
    end_scale = 1.5 * rtn_scale
    
    for i in range(101):
        scale = start_scale + (end_scale - start_scale) * i / 100.0
        for zero in range(-32, 33):
            q_list = []
            for val in x:
                q_val = round(val / scale) + zero
                if q_val < 0:
                    q_val = 0.0
                elif q_val > 15:
                    q_val = 15.0
                q_list.append(q_val)
            
            xhat_list = []
            for q_val in q_list:
                xhat_list.append(scale * (q_val - zero))
            
            obj = 0.0
            for idx in range(len(x)):
                diff = x[idx] - xhat_list[idx]
                if diff < 0:
                    diff = -diff
                obj += diff ** 0.7
                
            if obj < best_obj:
                best_obj = obj
                best = np.array(xhat_list, dtype=x.dtype)
                
    return best


def compare_4bit_quantizers(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    hqq = _hqq_reconstruct(x)
    rtn = _rtn_reconstruct(x)
    
    hqq_sum = 0.0
    rtn_sum = 0.0
    n = len(x)
    for i in range(n):
        hqq_diff = hqq[i] - x[i]
        rtn_diff = rtn[i] - x[i]
        hqq_sum += hqq_diff * hqq_diff
        rtn_sum += rtn_diff * rtn_diff
        
    return float(hqq_sum / n), float(rtn_sum / n)
