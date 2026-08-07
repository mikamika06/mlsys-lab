import ref
import numpy as np

def check(workdir):
    from pagedkv.batch import process_variable_batch
    m = {"variable_lens_ok": 0.0}
    seq_lens = [5, 20]
    res = process_variable_batch(seq_lens, 16)
    if res is not None:
        m["variable_lens_ok"] = 1.0
    return m
