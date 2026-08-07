import numpy as np
import ref


def check(workdir):
    from moe_balance.aux_loss import compute_switch_aux_loss, switch_grad_direction

    out = {"loss_matched": 0.0, "grad_direction_rel_err": 1.0}

    seq = ref.generate_synthetic_logits(num_batches=3, tokens=64, experts=4, seed=123)

    losses_match = True
    max_rel_err = 0.0

    for logits in seq:
        ref_loss, ref_grad = ref.compute_switch_aux_loss(logits, alpha=0.01)
        got_loss, got_grad = compute_switch_aux_loss(logits, alpha=0.01)
        direct_grad = switch_grad_direction(logits, alpha=0.01)

        if not np.isclose(ref_loss, got_loss, rtol=1e-6, atol=1e-6):
            losses_match = False

        err1 = np.linalg.norm(ref_grad - got_grad) / (np.linalg.norm(ref_grad) + 1e-12)
        err2 = np.linalg.norm(ref_grad - direct_grad) / (np.linalg.norm(ref_grad) + 1e-12)
        max_rel_err = max(max_rel_err, float(err1), float(err2))

    if losses_match:
        out["loss_matched"] = 1.0
    out["grad_direction_rel_err"] = float(max_rel_err)

    return out
