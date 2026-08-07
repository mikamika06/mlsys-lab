import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"unicode_samples_matched": 0.0, "round_trip_passed": 0.0}

    try:
        from bytefallback.convert import verify_round_trip
    except ImportError as e:
        out["_note"] = f"Import failure: {e}"
        return out

    vocab = ref.BASE_VOCAB
    matched_count = 0

    for sample in ref.RARE_UNICODE_SAMPLES:
        try:
            ok = verify_round_trip(sample, vocab)
            if ok:
                matched_count += 1
        except Exception as e:
            out["_note"] = f"verify_round_trip failed on {sample!r}: {e}"

    out["unicode_samples_matched"] = float(matched_count)
    if matched_count == len(ref.RARE_UNICODE_SAMPLES):
        out["round_trip_passed"] = 1.0

    return out
