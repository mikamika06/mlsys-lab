import numpy as np


def speculative_step(target_model, draft_model, input_ids):
    draft_out = draft_model(input_ids)
    target_out = target_model(input_ids)
    accepted = 0
    for d, t in zip(draft_out, target_out):
        if d == t:
            accepted += 1
        else:
            break
    return accepted, target_out
