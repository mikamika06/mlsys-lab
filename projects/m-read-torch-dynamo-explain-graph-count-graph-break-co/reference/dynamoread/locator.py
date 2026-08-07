import torch


def locate_break_line(fn, sample_args):
    explanation = torch._dynamo.explain(fn)(*sample_args)
    reasons = getattr(explanation, "break_reasons", [])
    if not reasons:
        return None
    return getattr(reasons[0], "user_stack", [None])[0]
