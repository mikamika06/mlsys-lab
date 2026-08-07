def pack_step(decodes, prefills, token_budget):
    raise NotImplementedError


def compute_steps(prefill_length, token_budget, decode_count):
    raise NotImplementedError


def predict_itl_jitter(prefill_length, token_budget, decode_count, base_decode_latency_ms):
    raise NotImplementedError
