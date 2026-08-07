import ref
import numpy as np

def check(workdir):
    from draftdistill.loss import combined_loss
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(10, 5))
    targets = rng.normal(size=(10, 5))
    f_pred = rng.normal(size=(10, 8))
    f_targ = rng.normal(size=(10, 8))

    got = combined_loss(logits, targets, f_pred, f_targ, alpha=0.4)
    want = ref.compute_combined_loss(logits, targets, f_pred, f_targ, alpha=0.4)

    rel_err = float(abs(got - want) / (abs(want) + 1e-8))
    out = {"loss_rel_err": rel_err}
    if rel_err > 0.01:
        out["_note"] = f"loss relative error {rel_err} exceeds 0.01 (got {got}, want {want})"
    return out
