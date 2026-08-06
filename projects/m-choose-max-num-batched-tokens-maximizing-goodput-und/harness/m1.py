import ref
from chunking.goodput import select_max_batched_tokens


def check(workdir):
    out = {"optimal_token_limit_matched": 0.0}
    workloads = ref.generate_workloads()
    candidates = [512, 1024, 2048, 4096, 8192]
    
    reference_val = ref.select_max_batched_tokens(workloads, candidates, itl_p99_limit_ms=40.0)
    learner_val = select_max_batched_tokens(workloads, candidates, itl_p99_limit_ms=40.0)
    
    if int(learner_val) == int(reference_val):
        out["optimal_token_limit_matched"] = 1.0
    else:
        out["_note"] = f"expected optimal token limit {reference_val}, got {learner_val}"
    return out
