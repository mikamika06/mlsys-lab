import math
import itertools
import numpy as np

def logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
    """Stable log‑sum‑exp over the given axis."""
    x = np.asarray(x, dtype=np.float64)
    ndim = len(x.shape)
    if axis is None:
        ranges = [range(s) for s in x.shape]
        m = -float('inf')
        for idx in itertools.product(*ranges):
            val = float(x[idx])
            if val > m:
                m = val
        sum_exp = 0.0
        for idx in itertools.product(*ranges):
            val = float(x[idx])
            sum_exp += math.exp(val - m)
        return np.float64(m + math.log(sum_exp))
    
    if isinstance(axis, int):
        axes = (axis % ndim,)
    elif isinstance(axis, tuple):
        axes = tuple(a % ndim for a in axis)
    else:
        axes = (int(axis) % ndim,)
        
    reduced_dims = set(axes)
    reduced_axes = [i for i in range(ndim) if i in reduced_dims]
    non_reduced_axes = [i for i in range(ndim) if i not in reduced_dims]
    
    out_shape = tuple(x.shape[i] for i in non_reduced_axes)
    out = np.empty(out_shape, dtype=np.float64)
    
    out_ranges = [range(x.shape[i]) for i in non_reduced_axes]
    red_ranges = [range(x.shape[i]) for i in reduced_axes]
    
    for out_idx in itertools.product(*out_ranges):
        full_idx_template = [0] * ndim
        for k, ax in enumerate(non_reduced_axes):
            full_idx_template[ax] = out_idx[k]
            
        m = -float('inf')
        for red_idx in itertools.product(*red_ranges):
            full_idx = list(full_idx_template)
            for k, ax in enumerate(reduced_axes):
                full_idx[ax] = red_idx[k]
            val = float(x[tuple(full_idx)])
            if val > m:
                m = val
                
        sum_exp = 0.0
        for red_idx in itertools.product(*red_ranges):
            full_idx = list(full_idx_template)
            for k, ax in enumerate(reduced_axes):
                full_idx[ax] = red_idx[k]
            val = float(x[tuple(full_idx)])
            sum_exp += math.exp(val - m)
            
        out[out_idx] = m + math.log(sum_exp)
        
    return out

def log_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """Stable log‑softmax over the given axis."""
    x = np.asarray(x, dtype=np.float64)
    ndim = len(x.shape)
    if axis is None:
        axes = tuple(range(ndim))
    elif isinstance(axis, int):
        axes = (axis % ndim,)
    elif isinstance(axis, tuple):
        axes = tuple(a % ndim for a in axis)
    else:
        axes = (int(axis) % ndim,)
        
    reduced_dims = set(axes)
    reduced_axes = [i for i in range(ndim) if i in reduced_dims]
    non_reduced_axes = [i for i in range(ndim) if i not in reduced_dims]
    
    out = np.empty(x.shape, dtype=np.float64)
    
    out_ranges = [range(x.shape[i]) for i in non_reduced_axes]
    red_ranges = [range(x.shape[i]) for i in reduced_axes]
    
    for out_idx in itertools.product(*out_ranges):
        full_idx_template = [0] * ndim
        for k, ax in enumerate(non_reduced_axes):
            full_idx_template[ax] = out_idx[k]
            
        m = -float('inf')
        for red_idx in itertools.product(*red_ranges):
            full_idx = list(full_idx_template)
            for k, ax in enumerate(reduced_axes):
                full_idx[ax] = red_idx[k]
            val = float(x[tuple(full_idx)])
            if val > m:
                m = val
                
        sum_exp = 0.0
        for red_idx in itertools.product(*red_ranges):
            full_idx = list(full_idx_template)
            for k, ax in enumerate(reduced_axes):
                full_idx[ax] = red_idx[k]
            val = float(x[tuple(full_idx)])
            sum_exp += math.exp(val - m)
            
        log_sum_exp = math.log(sum_exp)
        
        for red_idx in itertools.product(*red_ranges):
            full_idx = list(full_idx_template)
            for k, ax in enumerate(reduced_axes):
                full_idx[ax] = red_idx[k]
            t_idx = tuple(full_idx)
            out[t_idx] = float(x[t_idx]) - m - log_sum_exp
            
    return out
