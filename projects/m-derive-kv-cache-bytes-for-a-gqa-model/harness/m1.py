import ref

def check(workdir):
    from kvderive import calc
    ok = 0
    total = 0
    for cfg in ref.CONFIGS:
        for ctx in [512, 2048, 4096]:
            for dt in [2, 4]:
                total += 1
                want = ref.calc_kv_bytes(cfg, ctx, dt)
                try:
                    got = calc.calc_kv_bytes(cfg, ctx, dt)
                except Exception:
                    got = -1
                if got == want:
                    ok += 1
    matched = 1.0 if ok == total and total > 0 else 0.0
    return {"baseline_matched": matched}
