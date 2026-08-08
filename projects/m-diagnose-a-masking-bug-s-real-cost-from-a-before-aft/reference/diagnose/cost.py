def evaluate_cost(diff_metrics):
    inst_ratio = diff_metrics.get("mcu_inst_executed", {}).get("ratio", 1.0)
    stall_ratio = diff_metrics.get("stall_mio_throttle", {}).get("ratio", 1.0)
    reg_ratio = diff_metrics.get("registers_per_thread", {}).get("ratio", 1.0)

    score = (inst_ratio - 1.0) * 100.0 + (stall_ratio - 1.0) * 50.0 + (reg_ratio - 1.0) * 25.0
    is_regression = inst_ratio > 1.05 or stall_ratio > 1.05
    return {"score": score, "regression_detected": bool(is_regression)}
