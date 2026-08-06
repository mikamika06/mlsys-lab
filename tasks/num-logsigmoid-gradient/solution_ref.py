import math
import numpy as np


def logsigmoid_with_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    value_list = []
    grad_list = []
    for i in range(x.shape[0]):
        xi = x[i]
        if xi >= 0:
            val = -math.log1p(math.exp(-xi))
        else:
            val = xi - math.log1p(math.exp(xi))
        
        if xi >= 0:
            ex = math.exp(-xi)
            gr = ex / (1.0 + ex)
        else:
            ex = math.exp(xi)
            gr = 1.0 / (1.0 + ex)
            
        value_list.append(val)
        grad_list.append(gr)
        
    value = np.array(value_list, dtype=np.float64)
    grad = np.array(grad_list, dtype=np.float64)
    return value, grad
