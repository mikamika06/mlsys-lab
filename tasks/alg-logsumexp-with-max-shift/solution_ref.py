import math
import numpy as np

def logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Compute the log‑sum‑exp of `x` along `axis` with numerical stability."""
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        flat = x.ravel()
        max_val = flat[0]
        for i in range(1, len(flat)):
            if flat[i] > max_val:
                max_val = flat[i]
        sum_exp = 0.0
        for i in range(len(flat)):
            sum_exp += math.exp(flat[i] - max_val)
        return np.array(math.log(sum_exp) + max_val, dtype=np.float64)
    else:
        axis = axis % x.ndim
        out_shape = x.shape[:axis] + x.shape[axis + 1:]
        out = np.empty(out_shape, dtype=np.float64)
        out_flat = out.ravel()
        
        outer_size = 1
        for d in x.shape[:axis]:
            outer_size *= d
        
        axis_size = x.shape[axis]
        
        inner_size = 1
        for d in x.shape[axis + 1:]:
            inner_size *= d

        x_flat = x.ravel()

        for o in range(outer_size):
            for i in range(inner_size):
                base_idx = o * (axis_size * inner_size) + i
                max_val = x_flat[base_idx]
                for k in range(1, axis_size):
                    val = x_flat[base_idx + k * inner_size]
                    if val > max_val:
                        max_val = val
                
                sum_exp = 0.0
                for k in range(axis_size):
                    sum_exp += math.exp(x_flat[base_idx + k * inner_size] - max_val)
                
                out_flat[o * inner_size + i] = math.log(sum_exp) + max_val

        return out
