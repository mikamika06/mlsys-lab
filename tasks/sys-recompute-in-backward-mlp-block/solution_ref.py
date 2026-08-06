import numpy as np

def checkpoint_forward(x: np.ndarray,
                       W1: np.ndarray, b1: np.ndarray,
                       W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    h = W1.shape[0]
    o = W2.shape[0]
    d = x.shape[0]
    
    z = np.zeros(h, dtype=W1.dtype)
    for i in range(h):
        s = b1[i]
        for j in range(d):
            s += W1[i, j] * x[j]
        z[i] = s
        
    a = np.zeros(h, dtype=z.dtype)
    for i in range(h):
        if z[i] > 0:
            a[i] = z[i]
        else:
            a[i] = 0.0
            
    out = np.zeros(o, dtype=W2.dtype)
    for i in range(o):
        s = b2[i]
        for j in range(h):
            s += W2[i, j] * a[j]
        out[i] = s
        
    return out

def checkpoint_backward(dy: np.ndarray,
                        x: np.ndarray,
                        W1: np.ndarray, b1: np.ndarray,
                        W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    h = W1.shape[0]
    d = x.shape[0]
    o = W2.shape[0]
    
    z = np.zeros(h, dtype=W1.dtype)
    for i in range(h):
        s = b1[i]
        for j in range(d):
            s += W1[i, j] * x[j]
        z[i] = s
        
    mask = np.zeros(h, dtype=float)
    for i in range(h):
        if z[i] > 0:
            mask[i] = 1.0
        else:
            mask[i] = 0.0
            
    da = np.zeros(h, dtype=W2.dtype)
    for j in range(h):
        s = 0.0
        for i in range(o):
            s += W2[i, j] * dy[i]
        da[j] = s
        
    dz = np.zeros(h, dtype=da.dtype)
    for i in range(h):
        dz[i] = da[i] * mask[i]
        
    dx = np.zeros(d, dtype=W1.dtype)
    for j in range(d):
        s = 0.0
        for i in range(h):
            s += W1[i, j] * dz[i]
        dx[j] = s
        
    return dx
