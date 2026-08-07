import numpy as np
from harness import ref

def check(workdir):
    from lora_merge.merger import batch_verify_prompts
    bw, la, lb, alpha, rank, _ = ref.generate_test_data()
    merged = batch_verify_prompts(bw, la, lb, alpha, rank, num_prompts=200)
    return {"max_error_low": 1.0 if merged else 0.0}
