import time


def measure_stage_latencies(stage1_model, stage2_model, inputs, draft_steps):
    t0 = time.perf_counter()
    draft_tokens, s1_time = stage1_model.generate_draft(inputs, draft_steps)
    t1 = time.perf_counter()

    accepted_tokens, s2_time = stage2_model.verify_and_generate(inputs, draft_tokens)
    t2 = time.perf_counter()

    measured_total = (t1 - t0) + (t2 - t1)
    stage_sum = s1_time + s2_time
    total_latency = stage_sum if stage_sum > 0 else measured_total

    return {
        "stage1_latency": s1_time,
        "stage2_latency": s2_time,
        "total_latency": total_latency,
        "accepted_count": len(accepted_tokens),
        "draft_count": len(draft_tokens)
    }


def measure_single_stage_latency(target_model, inputs, total_tokens):
    t0 = time.perf_counter()
    output_tokens, execution_time = target_model.generate_baseline(inputs, total_tokens)
    t1 = time.perf_counter()

    total_latency = execution_time if execution_time > 0 else (t1 - t0)
    return {
        "total_latency": total_latency,
        "tokens_generated": len(output_tokens)
    }
