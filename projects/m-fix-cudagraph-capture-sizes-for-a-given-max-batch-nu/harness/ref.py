import random

def get_test_cases():
    cases = []
    for mb in [16, 32, 64]:
        for st in [2, 4, 5]:
            cases.append((mb, st))
    return cases

CANDIDATE_CONFIGS = [
    {"draft_tokens": 2, "acceptance_rate": 0.7, "kv_per_token_bytes": 512, "max_context_len": 2048, "max_batch_size": 8},
    {"draft_tokens": 4, "acceptance_rate": 0.6, "kv_per_token_bytes": 512, "max_context_len": 2048, "max_batch_size": 8},
    {"draft_tokens": 5, "acceptance_rate": 0.4, "kv_per_token_bytes": 512, "max_context_len": 2048, "max_batch_size": 8}
]

SAMPLE_LOG = """
[INFO] Initializing DraftEngine module...
[INFO] DraftEngine layers=24 hidden=2048 spec_tokens=4
[INFO] Allocation complete. Peak memory: 2048.0 MB
"""
