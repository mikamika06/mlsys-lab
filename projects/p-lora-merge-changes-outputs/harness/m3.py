import numpy as np
from harness import ref

def check(workdir):
    from lora_merge.merger import verify_scaling
    bw, la, lb, alpha, rank, x = ref.generate_test_data()
    res = verify_scaling(alpha, rank, la, lb)
    return {"scale_correct": 1.0 if res else 0.0}
