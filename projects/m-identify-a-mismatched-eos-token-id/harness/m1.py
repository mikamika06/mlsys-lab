import ref


def check(workdir):
    from gguf_meta.eos import find_mismatched_eos

    samples = ref.make_mock_metadata_samples()
    out = {"detected_mismatches": 0.0}

    results = [find_mismatched_eos(s) for s in samples]

    if (not results[0]["mismatch"]) and results[1]["mismatch"]:
        out["detected_mismatches"] = 1.0
    else:
        out["_note"] = f"Expected sample 0 mismatch=False, sample 1 mismatch=True. Got {results[0]} and {results[1]}"

    return out
