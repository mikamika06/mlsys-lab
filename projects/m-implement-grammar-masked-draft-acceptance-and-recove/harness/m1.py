import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from speculative.acceptance import compute_accepted_length
    except ImportError:
        return {"acceptance_rate": 0.0, "_note": "ImportError"}

    out = {"acceptance_rate": 0.0}
    ok = 0
    total = len(ref.BATCH_DRAFT_TOKENS)
    for i in range(total):
        want = ref.compute_accepted_length(
            ref.BATCH_DRAFT_TOKENS[i],
            ref.BATCH_TARGET_PROBS[i],
            ref.BATCH_DRAFT_PROBS[i],
            ref.BATCH_GRAMMAR_MASKS[i],
            ref.BATCH_RANDOM_SAMPLES[i]
        )
        try:
            got = compute_accepted_length(
                ref.BATCH_DRAFT_TOKENS[i],
                ref.BATCH_TARGET_PROBS[i],
                ref.BATCH_DRAFT_PROBS[i],
                ref.BATCH_GRAMMAR_MASKS[i],
                ref.BATCH_RANDOM_SAMPLES[i]
            )
            if got == want:
                ok += 1
        except Exception:
            pass

    out["acceptance_rate"] = float(ok) / total if total > 0 else 0.0
    return out
