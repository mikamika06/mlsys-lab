import numpy as np
from harness import ref

def check(workdir):
    from lora_merge.analyzer import detect_dtype_issue
    bw, la, lb, alpha, rank, x = ref.generate_test_data()
    res = detect_dtype_issue(bw, la, lb, alpha, rank)
    return {"dtype_bug_found": 1.0 if res else 0.0}
