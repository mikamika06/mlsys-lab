def quantize_weights(cfg):
    sym = cfg["symmetric"]
    w = cfg["weights"]
    mx = max(abs(x) for x in w)
    if sym:
        scale = mx / 7.0 if mx > 0 else 1.0
        q = [max(-8, min(7, round(val / scale))) for val in w]
        deq = [val * scale for val in q]
    else:
        mn = min(w)
        mx = max(w)
        rng = mx - mn if mx != mn else 1.0
        scale = rng / 15.0
        zp = round(-mn / scale)
        zp = max(0, min(15, zp))
        q = [max(0, min(15, round(val / scale) + zp)) for val in w]
        deq = [(val - zp) * scale for val in q]
    return {"quantized": q, "dequantized": deq, "symmetric": sym}
