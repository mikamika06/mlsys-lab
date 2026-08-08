import ref

def check(workdir):
    from amxlog.sweep import analyze_k_sweep
    records = ref.generate_sweep_records()
    got = analyze_k_sweep(records)
    want = {16: "avx2", 256: "avx512", 1024: "amx"}
    match = 1.0 if got == want else 0.0
    return {"sweep_match": match}
