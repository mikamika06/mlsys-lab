import ref

def check(workdir):
    from inductor_utils.parser import find_cpu_vectorization
    out = {"cpu_matches": 0.0, "configs": float(len(ref.CPU_SAMPLES))}
    ok = 0
    for i, sample in enumerate(ref.CPU_SAMPLES):
        got = find_cpu_vectorization(sample["code"])
        want = sample["expected"]
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"sample {i}: got {got}, want {want}"
    out["cpu_matches"] = float(ok)
    return out
