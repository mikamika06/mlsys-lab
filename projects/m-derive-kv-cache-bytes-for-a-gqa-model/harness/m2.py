import ref

def check(workdir):
    from kvderive import calc
    ok = 0
    total = 0
    for cfg in ref.CONFIGS:
        for ctx in [1024, 2048]:
            for qt in ["F32", "F16", "Q8_0", "Q4_0"]:
                total += 1
                want = ref.calc_quant_kv_bytes(cfg, ctx, qt)
                try:
                    got = calc.calc_quant_kv_bytes(cfg, ctx, qt)
                except Exception:
                    got = -1
                if got == want:
                    ok += 1
    matched = 1.0 if ok == total and total > 0 else 0.0
    return {"quant_matched": matched}
