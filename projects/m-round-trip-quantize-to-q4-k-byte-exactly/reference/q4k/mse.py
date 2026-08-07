import numpy as np
from q4k.quant import quantize_q4_k, dequantize_q4_k

def locate_dominating_subblock(x):
    x = np.asarray(x, dtype=np.float32).reshape(8, 32)
    b = quantize_q4_k(x.flatten())
    deq = dequantize_q4_k(b).reshape(8, 32)
    mse = np.mean((x - deq) ** 2, axis=1)
    return int(np.argmax(mse))
