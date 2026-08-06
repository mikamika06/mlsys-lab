import math
import numpy as np
from scipy.stats import norm


def nf4_levels() -> np.ndarray:
    """The 16 NF4 codebook levels: equal-probability-mass quantiles of N(0,1),
    asymmetric (8 non-negative incl. exact 0, 7 negative), normalized to [-1, 1]."""
    offset = 0.9677083
    
    lin_pos = []
    step_pos = (0.5 - offset) / (9 - 1)
    for i in range(9):
        lin_pos.append(offset + i * step_pos)
    v_pos_list = [norm.ppf(val) for val in lin_pos[:-1]]
    
    lin_neg = []
    step_neg = (0.5 - offset) / (8 - 1)
    for i in range(8):
        lin_neg.append(offset + i * step_neg)
    v_neg_list = [-norm.ppf(val) for val in lin_neg[:-1]]
    
    raw_v = v_pos_list + [0.0] + v_neg_list
    
    v_sorted = sorted(raw_v)
    
    max_abs = 0.0
    for val in v_sorted:
        aVal = abs(val)
        if aVal > max_abs:
            max_abs = aVal
            
    v_norm = [val / max_abs for val in v_sorted]
    
    out = np.zeros(16, dtype=np.float64)
    for i in range(16):
        out[i] = v_norm[i]
    return out


def quantize_4bit(x: np.ndarray, block_size: int = 64) -> tuple:
    """Blockwise absmax NF4 quantization: nearest-codebook-level 4-bit codes,
    packed two-per-byte, plus one float32 scale per block."""
    levels = nf4_levels()
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    n_blocks = math.ceil(n / block_size)

    codes = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(n_blocks, dtype=np.float32)
    
    for b in range(n_blocks):
        s = b * block_size
        e = min((b + 1) * block_size, n)
        
        am = 0.0
        for i in range(s, e):
            aVal = abs(x[i])
            if aVal > am:
                am = aVal
        am = float(am)
        if am == 0.0:
            am = 1.0
        absmax[b] = am
        
        for i in range(s, e):
            norm_val = x[i] / am
            best_idx = 0
            best_dist = abs(norm_val - levels[0])
            for l_idx in range(1, 16):
                dist = abs(norm_val - levels[l_idx])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = l_idx
            codes[i] = best_idx

    n_packed = (n + 1) // 2
    packed = np.zeros(n_packed, dtype=np.uint8)
    
    for i in range(n):
        c = codes[i]
        p_idx = i // 2
        if i % 2 == 0:
            packed[p_idx] |= c
        else:
            packed[p_idx] |= (c << 4)
            
    return packed, absmax


def dequantize_4bit(packed: np.ndarray, absmax: np.ndarray, n: int, block_size: int = 64) -> np.ndarray:
    """Inverse of quantize_4bit: unpack nibbles, look up codebook level, scale by block absmax."""
    levels = nf4_levels()
    packed = np.asarray(packed, dtype=np.uint8)
    
    codes = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        p_idx = i // 2
        byte_val = packed[p_idx]
        if i % 2 == 0:
            codes[i] = byte_val & 0x0F
        else:
            codes[i] = (byte_val >> 4) & 0x0F

    out = np.zeros(n, dtype=np.float64)
    n_blocks = math.ceil(n / block_size)
    for b in range(n_blocks):
        s = b * block_size
        e = min((b + 1) * block_size, n)
        scale = float(absmax[b])
        for i in range(s, e):
            out[i] = levels[codes[i]] * scale
            
    return out
