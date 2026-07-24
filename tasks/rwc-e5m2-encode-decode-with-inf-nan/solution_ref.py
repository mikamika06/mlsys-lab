import numpy as np


def encode_e5m2(values: np.ndarray) -> np.ndarray:
    """Encode float32 array to E5M2 uint8 codes."""
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.uint8)
    
    u32 = np.frombuffer(values.tobytes(), dtype=np.uint32).reshape(values.shape)
    sign_bits = (u32 >> 31).astype(np.uint8)
    
    flat = values.ravel()
    out_flat = out.ravel()
    sign_flat = sign_bits.ravel()
    
    for idx in range(len(flat)):
        v = float(flat[idx])
        s = int(sign_flat[idx])
        
        if np.isnan(v):
            out_flat[idx] = (s << 7) | 0x7F
            continue
        
        av = abs(v)
        
        if np.isinf(v) or av > 57344.0:
            out_flat[idx] = (s << 7) | 0x7C
            continue
        
        if av == 0.0:
            out_flat[idx] = (s << 7)
            continue
        
        import math
        e_unbiased = int(math.floor(math.log2(av)))
        e_stored = e_unbiased + 15
        
        if e_stored <= 0:
            scale = 2.0 ** (-14) / 4.0
            m_exact = av / scale
            m_floor = int(m_exact)
            frac = m_exact - m_floor
            if frac > 0.5:
                m_round = m_floor + 1
            elif frac == 0.5:
                m_round = m_floor + 1 if (m_floor % 2 == 1) else m_floor
            else:
                m_round = m_floor
            m_round = min(m_round, 3)
            out_flat[idx] = (s << 7) | m_round
        else:
            e_stored = max(1, e_stored)
            if e_stored > 30:
                out_flat[idx] = (s << 7) | 0x7C
                continue
            scale = 2.0 ** (e_stored - 15)
            significand = av / scale
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
                e_stored += 1
                m_round = 0
                if e_stored > 30:
                    out_flat[idx] = (s << 7) | 0x7C
                    continue
            
            out_flat[idx] = (s << 7) | (e_stored << 2) | m_round
    
    return out


def decode_e5m2(codes: np.ndarray) -> np.ndarray:
    """Decode E5M2 uint8 codes to float32."""
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
            flat_o[idx] = sign * (2.0 ** (-14)) * (m / 4.0)
        else:
            flat_o[idx] = sign * (2.0 ** (e - 15)) * (1.0 + m / 4.0)
    
    return out
