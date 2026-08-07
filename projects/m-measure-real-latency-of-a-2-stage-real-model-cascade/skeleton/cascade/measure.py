def measure_stage_latencies(stage1_model, stage2_model, inputs, draft_steps):
    """
    Measures per-stage latency and accepted token count for a 2-stage model cascade.
    """
    raise NotImplementedError


def measure_single_stage_latency(target_model, inputs, total_tokens):
    """
    Measures total latency for a single-stage baseline generating total_tokens.
    """
    raise NotImplementedError
