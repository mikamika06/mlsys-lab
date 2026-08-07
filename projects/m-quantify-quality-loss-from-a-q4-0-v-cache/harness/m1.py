import ref


def check(workdir):
    from vcache.slot_memory import predict_multi_slot_growth

    out = {"slots_matched": 0.0}
    want = ref.predict_multi_slot_growth(ref.CONFIGS)
    try:
        got = predict_multi_slot_growth(ref.CONFIGS)
    except Exception as e:
        out["_note"] = f"predict_multi_slot_growth raised {type(e).__name__}: {e}"
        return out

    matched = sum(1 for w, g in zip(want, got) if w == g)
    out["slots_matched"] = float(matched)
    if matched < len(want):
        out["_note"] = f"Matched {matched}/{len(want)}. Want {want[:2]}, got {got[:2]}"
    return out
