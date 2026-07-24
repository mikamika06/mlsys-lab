def fused_elementwise_speedup(x, y):
    z = x * y
    out = z + 1
    unfused_bytes = x.nbytes + y.nbytes + z.nbytes + z.nbytes + out.nbytes
    fused_bytes = x.nbytes + y.nbytes + out.nbytes
    return float(unfused_bytes / fused_bytes)
