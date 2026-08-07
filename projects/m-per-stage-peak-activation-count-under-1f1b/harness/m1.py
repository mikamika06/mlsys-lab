import ref

def check(workdir):
    from pact.peak import compute_peak_activations
    out = {"peaks_matched": 0.0, "configs": float(len(ref.TEST_CASES))}
    ok = 0
    for i, (P, M) in enumerate(ref.TEST_CASES):
        want = ref.compute_peak_activations(P, M)
        got = compute_peak_activations(P, M)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case P={P}, M={M}: got {got}, reference {want}"
    out["peaks_matched"] = 1.0 if ok == len(ref.TEST_CASES) else 0.0
    return out
