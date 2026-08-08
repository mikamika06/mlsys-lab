import numpy as np

def generate_tensors():
    np.random.seed(42)
    t1 = np.random.randn(100).astype(np.float32)
    t2 = np.random.randn(50).astype(np.float32) * 5 + 10
    t2[0] = 150.0
    t3 = np.linspace(-5.0, 5.0, 200).astype(np.float32)
    t4 = np.random.rand(1000).astype(np.float32) * 2 - 1
    t4[500] = -50.0
    return [t1, t2, t3, t4]

TENSORS = generate_tensors()
SCHEMES = ["int8-sym", "uint4-asym", "int16-sym", "uint8-asym"]

def parse_scheme(name):
    parts = name.split("-")
    dtype = parts[0]
    sym = parts[1] == "sym"
    if dtype.startswith("int"):
        bits = int(dtype[3:])
    else:
        bits = int(dtype[4:])
    return {"bits": bits, "symmetric": sym}

def minmax_observer(x, args):
    bits = args["bits"]
    sym = args["symmetric"]
    if sym:
        qmin = -(2**(bits-1))
        qmax = 2**(bits-1) - 1
        m = float(np.max(np.abs(x)))
        if m == 0:
            return 1.0, 0.0
        scale = m / qmax
        zp = 0.0
    else:
        qmin = 0
        qmax = 2**bits - 1
        m_min, m_max = float(np.min(x)), float(np.max(x))
        if m_min == m_max:
            return 1.0, 0.0
        scale = (m_max - m_min) / qmax
        zp = np.clip(np.round(-m_min / scale), qmin, qmax)
    return float(scale), float(zp)

def mse_observer(x, args):
    b_scale, b_zp = minmax_observer(x, args)
    if b_scale == 1.0 and (np.max(x) == np.min(x)):
        return 1.0, 0.0

    alphas = np.linspace(0.1, 1.0, 100)
    best_mse = float('inf')
    best_scale = b_scale

    bits = args["bits"]
    sym = args["symmetric"]
    qmin = -(2**(bits-1)) if sym else 0
    qmax = 2**(bits-1) - 1 if sym else 2**bits - 1

    for a in alphas:
        s = b_scale * a
        xq = np.clip(np.round(x / s) + b_zp, qmin, qmax)
        x_approx = (xq - b_zp) * s
        mse = np.mean((x - x_approx)**2)
        if mse < best_mse:
            best_mse = mse
            best_scale = s

    return float(best_scale), float(b_zp)

def ignored_zp_bias(x, args, method):
    if method == "mse":
        scale, zp = mse_observer(x, args)
    else:
        scale, zp = minmax_observer(x, args)

    bits = args["bits"]
    sym = args["symmetric"]
    qmin = -(2**(bits-1)) if sym else 0
    qmax = 2**(bits-1) - 1 if sym else 2**bits - 1

    xq = np.clip(np.round(x / scale) + zp, qmin, qmax)
    x_correct = (xq - zp) * scale
    x_ignored = xq * scale
    return float(np.sum(x_ignored) - np.sum(x_correct))
