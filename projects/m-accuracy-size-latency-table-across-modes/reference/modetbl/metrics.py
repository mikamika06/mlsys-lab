def compute_size_ratios(profiles):
    fp32_size = next(p["size_bytes"] for p in profiles if p["mode"] == "FP32")
    out = {}
    for p in profiles:
        out[p["mode"]] = round(p["size_bytes"] / fp32_size, 4)
    return out


def evaluate_tradeoffs(profiles):
    ratios = compute_size_ratios(profiles)
    valid = True
    for p in profiles:
        if p["mode"] != "FP32" and ratios[p["mode"]] >= 1.0:
            valid = False
    return {"valid": valid, "ratios": ratios}
