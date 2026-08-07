def analyze_execution_metrics(prefill_stats, decode_stats):
    p_flops = float(prefill_stats.get("prefill_flops", 0))
    d_p_flops = float(decode_stats.get("prefill_flops", 0))
    d_d_flops = float(decode_stats.get("decode_flops", 0))

    total_decode = d_p_flops + d_d_flops
    ratio = (d_p_flops / total_decode) if total_decode > 0 else 0.0

    return {
        "prefill_worker_flops": p_flops,
        "decode_prefill_flops": d_p_flops,
        "decode_generation_flops": d_d_flops,
        "decode_prefill_ratio": ratio,
        "decode_prefill_steps": int(decode_stats.get("prefill_steps", 0))
    }


def verify_decode_skips_prefill(pipeline_result):
    p_stats = pipeline_result["prefill_stats"]
    d_stats = pipeline_result["decode_stats"]
    analysis = analyze_execution_metrics(p_stats, d_stats)

    no_decode_prefill_steps = (analysis["decode_prefill_steps"] == 0)
    low_prefill_flops = (analysis["decode_prefill_ratio"] <= 0.05)

    return {
        "verified": bool(no_decode_prefill_steps and low_prefill_flops),
        "analysis": analysis
    }
