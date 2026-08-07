def check_autocast_promotion(ops_list):
    promoted = []
    for op in ops_list:
        if op.get("precision") == "fp16" and op.get("sensitive") is False:
            promoted.append(op["name"])
    return promoted


def analyze_scaler_step(scaler_state, has_inf):
    scale = scaler_state["scale"]
    if has_inf:
        scale = scale * scaler_state["backoff_factor"]
        skipped = True
    else:
        scale = scale * scaler_state["growth_factor"]
        skipped = False
    return {"scale": scale, "skipped": skipped}
