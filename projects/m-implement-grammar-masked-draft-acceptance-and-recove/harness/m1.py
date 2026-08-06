import ref
import numpy as np


def check(workdir):
    from speculative.acceptance import compute_accepted_length

    unmasked_drafts, target_probs_list, grammar_masks = ref.generate_fixtures()
    ref_lengths = []
    for dt, tp, gm in zip(unmasked_drafts, target_probs_list, grammar_masks):
        dp = [np.ones((ref.VOCAB_SIZE,)) / ref.VOCAB_SIZE for _ in dt]
        ref_lengths.append(ref.compute_accepted_length(dt, tp, dp, gm))

    got_lengths = []
    for dt, tp, gm in zip(unmasked_drafts, target_probs_list, grammar_masks):
        dp = [np.ones((ref.VOCAB_SIZE,)) / ref.VOCAB_SIZE for _ in dt]
        got_lengths.append(compute_accepted_length(dt, tp, dp, gm))

    matches = sum(1 for r, g in zip(ref_lengths, got_lengths) if r == g)
    rate = float(matches) / float(len(ref_lengths))
    return {"acceptance_rate": rate}
