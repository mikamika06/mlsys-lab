import numpy as np

def derive_bias_scale(input_scale, weight_scales):
    return (input_scale * weight_scales).astype(np.float32)

def dequantize_weights(w_q, w_scales):
    return w_q.astype(np.float32) * w_scales[:, None, None, None]

def quantize_bias(b_real, b_scales):
    return np.round(b_real / b_scales).astype(np.int32)

def integer_conv2d(i_q, i_z, w_q, b_q):
    _, H, W, Cin = i_q.shape
    Cout, Kh, Kw, _ = w_q.shape
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    out = np.zeros((1, H_out, W_out, Cout), dtype=np.int32)
    i_shifted = i_q.astype(np.int32) - i_z
    w_int = w_q.astype(np.int32)
    for y in range(H_out):
        for x in range(W_out):
            patch = i_shifted[:, y:y+Kh, x:x+Kw, :]
            out[:, y, x, :] = np.tensordot(patch, w_int, axes=([1, 2, 3], [1, 2, 3])) + b_q
    return out
    
def generate_fixtures():
    np.random.seed(42)
    Cout, Cin = 8, 4
    Kh, Kw = 3, 3
    H, W = 10, 10
    
    i_scale = 0.5
    i_z = 128
    i_q = np.random.randint(0, 255, size=(1, H, W, Cin)).astype(np.uint8)
    w_scales = np.random.uniform(0.01, 0.1, size=(Cout,)).astype(np.float32)
    w_q = np.random.randint(-127, 127, size=(Cout, Kh, Kw, Cin)).astype(np.int8)
    b_real = np.random.uniform(-2.0, 2.0, size=(Cout,)).astype(np.float32)
    
    return i_scale, i_z, i_q, w_scales, w_q, b_real
