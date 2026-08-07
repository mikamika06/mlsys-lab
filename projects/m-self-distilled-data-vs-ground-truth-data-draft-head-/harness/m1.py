import numpy as np

def check(workdir):
    from draftdistill.evaluation import compute_acceptance_rate
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(50, 10))
    tokens = rng.integers(0, 10, size=(50,))
    got = compute_acceptance_rate(logits, tokens)

    preds = np.argmax(logits, axis=-1)
    want = float(np.mean(preds == tokens))

    err = float(abs(got - want))
    out = {"acceptance_delta_match": 1.0 if err < 1e-5 else 0.0}
    if err >= 1e-5:
        out["_note"] = f"expected acceptance rate {want}, got {got}"
    return out
