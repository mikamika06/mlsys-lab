def memory_error_analysis():
    return {
        "bf16_memory_savings": "equal to fp16, but dynamic range is much higher preventing underflow/overflow",
        "manual_half_pitfalls": "lacks automatic scaling and promotion, prone to underflow in gradients"
    }
