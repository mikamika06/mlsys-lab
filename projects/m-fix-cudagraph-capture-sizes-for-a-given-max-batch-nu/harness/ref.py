CONFIGS = [
    {"max_batch_size": 16, "num_speculative_tokens": 3},
    {"max_batch_size": 32, "num_speculative_tokens": 1},
    {"max_batch_size": 8, "num_speculative_tokens": 5},
]

EAGLE_CONFIGS = [
    {"name": "cfg_low", "kv_bytes": 1024 * 1024, "throughput_score": 15.0},
    {"name": "cfg_mid", "kv_bytes": 2048 * 1024, "throughput_score": 25.0},
    {"name": "cfg_high", "kv_bytes": 4096 * 1024, "throughput_score": 30.0},
]

SAMPLE_LOG = """
[TRT-LLM] Initializing build...
Draft engine avg latency: 5.2 ms
Draft acceptance rate: 0.78
Draft engine peak memory: 5242880 bytes
[TRT-LLM] Build complete.
"""


def compute_capture_sizes(max_batch_size, num_speculative_tokens):
    from reference.speculative.cudagraph import compute_capture_sizes as ref_fn
    return ref_fn(max_batch_size, num_speculative_tokens)


def find_optimal_eagle_config(configs, budget):
    from reference.speculative.eagle import find_optimal_eagle_config as ref_fn
    return ref_fn(configs, budget)


def extract_draft_engine_stats(log):
    from reference.speculative.trtlog import extract_draft_engine_stats as ref_fn
    return ref_fn(log)
