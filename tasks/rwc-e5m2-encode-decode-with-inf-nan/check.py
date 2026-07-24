import numpy as np


def _ref_encode_e5m2(values: np.ndarray) -> np.ndarray:
    """Reference E5M2 encoder (pure numpy, RNE)."""
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.uint8)
    
    sign_bits = (np.frombuffer(values.tobytes(), dtype=np.uint32).reshape(values.shape) >> 31).astype(np.uint8)
    
    flat = values.ravel()
    out_flat = out.ravel()
    sign_flat = sign_bits.ravel()
    
    for idx in range(len(flat)):
        v = float(flat[idx])
        s = sign_flat[idx]
        
        if np.isnan(v):
            # NaN: exp=31, mantissa=3 (0b11)
            out_flat[idx] = (s << 7) | 0x7F
            continue
        
        av = abs(v)
        
        if np.isinf(v) or av > 57344.0:
            # infinity or overflow
            out_flat[idx] = (s << 7) | 0x7C  # exp=31, mantissa=0
            continue
        
        if av == 0.0:
            out_flat[idx] = (s << 7)
            continue
        
        # Find exponent
        e_unbiased = int(np.floor(np.log2(av))) if av >= 1.0 else int(np.floor(np.log2(av)))
        # clamp to subnormal range
        e_stored = e_unbiased + 15  # bias=15
        
        if e_stored <= 0:
            # subnormal
            # value = sign * 2^(-14) * (m/4)
            # m = round(value / 2^(-14) * 4)
            scale = 2.0 ** (-14) / 4.0
            m_exact = av / scale
            # RNE
            m_floor = int(m_exact)
            frac = m_exact - m_floor
            if frac > 0.5:
                m_round = m_floor + 1
            elif frac == 0.5:
                m_round = m_floor + 1 if (m_floor % 2 == 1) else m_floor
            else:
                m_round = m_floor
            m_round = min(m_round, 3)
            if m_round == 0 and m_floor == 0 and frac < 0.5:
                out_flat[idx] = (s << 7)
            else:
                out_flat[idx] = (s << 7) | (0 << 2) | m_round
        else:
            e_stored = max(1, e_stored)
            if e_stored > 30:
                # overflow to inf
                out_flat[idx] = (s << 7) | 0x7C
                continue
            # normal
            # value = 2^(e_stored-15) * (1 + m/4)
            scale = 2.0 ** (e_stored - 15)
            significand = av / scale  # should be in [1, 2)
            m_exact = (significand - 1.0) * 4.0
            m_floor = int(m_exact)
            frac = m_exact - m_floor
            if frac > 0.5:
                m_round = m_floor + 1
            elif frac == 0.5:
                m_round = m_floor + 1 if (m_floor % 2 == 1) else m_floor
            else:
                m_round = m_floor
            
            if m_round >= 4:
                # carry into exponent
                e_stored += 1
                m_round = 0
                if e_stored > 30:
                    out_flat[idx] = (s << 7) | 0x7C
                    continue
            
            out_flat[idx] = (s << 7) | (e_stored << 2) | m_round
    
    return out


def _ref_decode_e5m2(codes: np.ndarray) -> np.ndarray:
    """Reference E5M2 decoder."""
    codes = np.asarray(codes, dtype=np.uint8)
    out = np.zeros(codes.shape, dtype=np.float32)
    flat_c = codes.ravel()
    flat_o = out.ravel()
    
    for idx in range(len(flat_c)):
        c = int(flat_c[idx])
        s = (c >> 7) & 1
        e = (c >> 2) & 0x1F
        m = c & 0x3
        sign = -1.0 if s else 1.0
        
        if e == 31:
            if m == 0:
                flat_o[idx] = sign * np.inf
            else:
                flat_o[idx] = np.nan
        elif e == 0:
            # subnormal
            flat_o[idx] = sign * (2.0 ** (-14)) * (m / 4.0)
        else:
            flat_o[idx] = sign * (2.0 ** (e - 15)) * (1.0 + m / 4.0)
    
    return out


def grade(sol, fx) -> dict:
    # Test values: special cases + normal values
    test_values = np.array([
        0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
        0.5, 0.25, 0.125,
        57344.0, -57344.0,   # max finite
        1e6, -1e6,           # overflow -> inf
        np.inf, -np.inf,     # inf
        np.nan,              # nan
        1.5, 3.0, 0.0625,
        6.0, 12.0, 0.001953125,  # 2^(-9)
    ], dtype=np.float32)
    
    ref_codes = _ref_encode_e5m2(test_values)
    ref_decoded = _ref_decode_e5m2(ref_codes)
    
    try:
        sol_codes = np.asarray(sol.encode_e5m2(test_values.copy()), dtype=np.uint8)
        sol_decoded = np.asarray(sol.decode_e5m2(sol_codes.copy()), dtype=np.float32)
    except Exception:
        return {"exact_match": 0.0}
    
    if sol_codes.shape != ref_codes.shape:
        return {"exact_match": 0.0}
    
    # Compare codes (byte exact), handling NaN positions specially
    nan_mask = np.isnan(test_values)
    # For nan inputs: just check that decoded result is also NaN
    # For non-nan: check exact byte match
    codes_match = np.all(sol_codes[~nan_mask] == ref_codes[~nan_mask])
    
    # Check decoded values
    ref_nan_pos = np.isnan(ref_decoded)
    sol_nan_pos = np.isnan(sol_decoded)
    nan_positions_match = np.all(ref_nan_pos == sol_nan_pos)
    
    non_nan = ~ref_nan_pos
    vals_match = np.all(sol_decoded[non_nan] == ref_decoded[non_nan])
    
    exact = float(codes_match and nan_positions_match and vals_match)
    return {"exact_match": exact}
