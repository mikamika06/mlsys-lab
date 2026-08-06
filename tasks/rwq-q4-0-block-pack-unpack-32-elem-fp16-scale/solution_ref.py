import numpy as np


def q4_0_block_pack_unpack(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    blocks = x.reshape(-1, 32)
    B = blocks.shape[0]

    scale_list = []
    for i in range(B):
        max_val = -1.0
        max_signed = 0.0
        for j in range(32):
            val = blocks[i, j]
            abs_val = val if val >= 0.0 else -val
            if j == 0 or abs_val > max_val:
                max_val = abs_val
                max_signed = val
        d_val = max_signed / -8.0
        scale_list.append(d_val)

    d16 = np.array(scale_list, dtype=np.float16)
    dq = d16.astype(np.float64)

    nibbles_list = []
    dequant_list = []
    for i in range(B):
        dq_val = dq[i]
        safe_dq_val = 1.0 if dq_val == 0.0 else dq_val
        block_nibbles = []
        block_dequant = []
        for j in range(32):
            val = blocks[i, j]
            if dq_val == 0.0:
                nibble_val = 8
            else:
                q_val = round(val / safe_dq_val) + 8.0
                if q_val < 0.0:
                    q_val = 0.0
                elif q_val > 15.0:
                    q_val = 15.0
                nibble_val = int(q_val)
            block_nibbles.append(nibble_val)
            deq_val = (float(nibble_val) - 8.0) * dq_val
            block_dequant.append(deq_val)
        nibbles_list.append(block_nibbles)
        dequant_list.append(block_dequant)

    nibbles = np.array(nibbles_list, dtype=np.uint8)
    dequant = np.array(dequant_list, dtype=np.float64)

    return {"scale": d16, "nibbles": nibbles, "dequant": dequant}
