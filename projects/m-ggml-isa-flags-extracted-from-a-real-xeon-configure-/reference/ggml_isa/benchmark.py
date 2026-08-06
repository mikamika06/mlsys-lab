def compare_performance(amx_tps, avx_tps):
    speedup = amx_tps / max(avx_tps, 1e-6)
    return {"speedup": float(speedup), "efficient": bool(speedup > 1.1)}
