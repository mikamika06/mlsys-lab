import numpy as np
from harness import ref

def check(workdir):
    from lora_merge.merger import safe_merge
    bw, la, lb, alpha, rank, x = ref.generate_test_data()
    merged = safe_merge(bw, la, lb, alpha, rank)
    out_adapter = (x @ bw.T) + (x @ ((lb @ la) * (alpha / rank)).T)
    out_merged = x @ merged.T
    err = np.max(np.abs(out_adapter - out_merged))
    return {"merge_safe": 1.0 if err < 1e-5 else 0.0}
