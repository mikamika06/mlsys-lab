import numpy as np

def unpack_scales_and_mins(packed_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    assert len(packed_bytes) == 12
    s_bytes = packed_bytes[:6]
    m_bytes = packed_bytes[6:]
    u64_s = int.from_bytes(s_bytes, byteorder='little')
    scales = np.array([(u64_s >> (6 * i)) & 0x3F for i in range(8)], dtype=np.float32)
    u64_m = int.from_bytes(m_bytes, byteorder='little')
    mins = np.array([(u64_m >> (6 * i)) & 0x3F for i in range(8)], dtype=np.float32)
    return scales, mins
