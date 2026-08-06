import math
import itertools
import numpy as np


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    out = np.empty(shape, dtype=np.float64)
    
    if len(shape) == 0:
        return np.array(0.0, dtype=np.float64)
    
    prefix_shape = shape[:-1]
    last_dim = shape[-1]
    
    if not prefix_shape:
        m = x[0]
        for i in range(1, last_dim):
            if x[i] > m:
                m = x[i]
        
        exp_sum = 0.0
        for i in range(last_dim):
            exp_sum += math.exp(x[i] - m)
        
        log_exp_sum = math.log(exp_sum)
        for i in range(last_dim):
            out[i] = (x[i] - m) - log_exp_sum
    else:
        for p in itertools.product(*(range(d) for d in prefix_shape)):
            m = x[p + (0,)]
            for i in range(1, last_dim):
                val = x[p + (i,)]
                if val > m:
                    m = val
            
            exp_sum = 0.0
            for i in range(last_dim):
                exp_sum += math.exp(x[p + (i,)] - m)
            
            log_exp_sum = math.log(exp_sum)
            for i in range(last_dim):
                out[p + (i,)] = (x[p + (i,)] - m) - log_exp_sum
                
    return out


def log_softmax_vjp(x, g):
    """Vector-Jacobian product of y = log_softmax(x, axis=-1).

    Given the upstream gradient `g` (dLoss/dy, same shape as x), returns
    dLoss/dx = g - softmax(x) * sum(g, axis=-1, keepdims=True).
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    shape = x.shape
    out = np.empty(shape, dtype=np.float64)
    
    log_s = _log_softmax(x)
    
    if len(shape) == 0:
        return np.array(g.item() - math.exp(log_s.item()) * g.item(), dtype=np.float64)
    
    prefix_shape = shape[:-1]
    last_dim = shape[-1]
    
    if not prefix_shape:
        softmax_arr = [math.exp(log_s[i]) for i in range(last_dim)]
        g_sum = 0.0
        for i in range(last_dim):
            g_sum += g[i]
        
        for i in range(last_dim):
            out[i] = g[i] - softmax_arr[i] * g_sum
    else:
        for p in itertools.product(*(range(d) for d in prefix_shape)):
            g_sum = 0.0
            for i in range(last_dim):
                g_sum += g[p + (i,)]
            
            for i in range(last_dim):
                sm = math.exp(log_s[p + (i,)])
                out[p + (i,)] = g[p + (i,)] - sm * g_sum
                
    return out
