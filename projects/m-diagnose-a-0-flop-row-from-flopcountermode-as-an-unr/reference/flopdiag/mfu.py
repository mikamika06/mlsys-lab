def rank_mfu(records):
    mfus = []
    for r in records:
        tokens_per_sec = r["tokens_per_sec"]
        params = r["params"]
        peak_tflops = r["peak_tflops"]
        flops_per_token = 6 * params
        achieved_tflops = (tokens_per_sec * flops_per_token) / 1e12
        mfu = achieved_tflops / peak_tflops
        mfus.append((r["id"], mfu))
    mfus.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in mfus]
