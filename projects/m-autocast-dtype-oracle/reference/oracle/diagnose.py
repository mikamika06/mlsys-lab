def diagnose_fp32_regions(trace_result):
    """Diagnose why specific operations or regions remained in FP32."""
    diagnoses = []
    for item in trace_result.get("intermediates", []):
        if item["actual_dtype"] == "float32":
            op = item["op"]
            reason = item["reason"]
            if reason == "op_requires_fp32":
                explanation = f"Operation '{op}' is registered as strictly FP32 for numerical stability."
            elif reason == "explicit_cast":
                explanation = f"Operation '{op}' has an explicit target_dtype override to float32."
            elif reason == "promoted_from_fp32_input":
                explanation = f"Operation '{op}' was promoted to float32 due to FP32 input operands."
            else:
                explanation = f"Operation '{op}' stayed in float32 due to default fallback rules."
            
            diagnoses.append({
                "out": item["out"],
                "op": op,
                "dtype": "float32",
                "reason": reason,
                "explanation": explanation
            })
    return diagnoses
