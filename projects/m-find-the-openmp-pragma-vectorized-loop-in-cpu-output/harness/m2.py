import ref

def check(workdir):
    from inductor_utils.metrics import extract_triton_metadata
    out = {"metadata_matches": 0.0, "configs": float(len(ref.TRITON_SAMPLES))}
    ok = 0
    for i, sample in enumerate(ref.TRITON_SAMPLES):
        got = extract_triton_metadata(sample["dump"])
        want = sample["expected"]
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"sample {i}: got {got}, want {want}"
    out["metadata_matches"] = float(ok)
    return out
