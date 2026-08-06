def check_fp8_availability(cfg):
    hd = cfg["head_dim"]
    dt = cfg["dtype"]
    if dt != "fp8":
        return {"available": True, "reason": "not_fp8"}
    if hd % 16 != 0:
        return {"available": False, "reason": "alignment_not_multiple_of_16"}
    if hd > 128:
        return {"available": False, "reason": "head_dim_exceeds_fp8_limit"}
    return {"available": True, "reason": "supported"}
