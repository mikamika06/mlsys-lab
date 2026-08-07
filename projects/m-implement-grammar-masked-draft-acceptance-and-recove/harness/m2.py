import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from speculative.metrics import measure_acceptance_loss
    except ImportError:
        return {"loss_match": 0.0, "_note": "ImportError"}

    out = {"loss_match": 0.0}
    want = ref.measure_acceptance_loss(
        ref.BATCH_DRAFT_TOKENS,
        ref.BATCH_TARGET_PROBS,
        ref.BATCH_DRAFT_PROBS,
        ref.BATCH_GRAMMAR_MASKS,
        ref.BATCH_RANDOM_SAMPLES
    )

    try:
        got = measure_acceptance_loss(
            ref.BATCH_DRAFT_TOKENS,
            ref.BATCH_TARGET_PROBS,
            ref.BATCH_DRAFT_PROBS,
            ref.BATCH_GRAMMAR_MASKS,
            ref.BATCH_RANDOM_SAMPLES
        )
        if abs(got - want) < 1e-5:
            out["loss_match"] = 1.0
    except Exception:
        pass

    return out
