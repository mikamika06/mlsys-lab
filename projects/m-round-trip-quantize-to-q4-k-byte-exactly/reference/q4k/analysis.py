import numpy as np
from q4k.quant import pack_q4_k, unpack_q4_k

def find_worst_subblock(weights):
    w = np.asarray(weights, dtype=np.float32)
    packed = pack_q4_k(w)
    dequant = unpack_q4_k(packed, len(w))
    sub_orig = w.reshape(8, 32)
    sub_deq = dequant.reshape(8, 32)
    mses = np.mean((sub_orig - sub_deq) ** 2, axis=1)
    return int(np.argmax(mses))
