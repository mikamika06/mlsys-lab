import ref


def check(workdir):
    from tile_recon.classify import classify_errors
    out = {"accuracy": 0.0}
    try:
        got = classify_errors(ref.ERROR_STRINGS)
        want = ref.EXPECTED_CLASSES
        if got == want:
            out["accuracy"] = 1.0
        else:
            correct = sum(1 for g, w in zip(got, want) if g == w)
            out["accuracy"] = float(correct / len(want))
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}"
    return out
