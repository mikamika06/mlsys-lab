import ref
import numpy as np


def check(workdir):
    from eagle.sampler import DraftSampler
    m = {"sampling_match": 0.0}
    try:
        sampler = DraftSampler(temperature=1.0)
        logits = [0.1, 2.5, 0.3, 1.2]
        res = sampler.sample(logits)
        if res == 1:
            m["sampling_match"] = 1.0
    except Exception:
        pass
    return m
