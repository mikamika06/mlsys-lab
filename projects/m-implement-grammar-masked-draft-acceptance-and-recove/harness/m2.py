import ref
from reference.speculative.metrics import measure_acceptance_loss as ref_measure


def check(workdir):
    from speculative.metrics import measure_acceptance_loss

    unmasked_drafts, target_probs_list, grammar_masks = ref.generate_fixtures()
    ref_res = ref_measure(unmasked_drafts, target_probs_list, grammar_masks)
    try:
        got_res = measure_acceptance_loss(unmasked_drafts, target_probs_list, grammar_masks)
    except Exception:
        return {"loss_match": 0.0}

    if isinstance(got_res, dict) and abs(got_res.get("acceptance_loss", -1) - ref_res["acceptance_loss"]) < 1e-5:
        return {"loss_match": 1.0}
    return {"loss_match": 0.0}
