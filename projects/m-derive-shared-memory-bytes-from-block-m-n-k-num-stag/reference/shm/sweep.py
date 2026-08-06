def analyze_sweep(records: list) -> dict:
    best_warp = None
    max_occ = -1.0
    for r in records:
        occ = r.get("achieved_occupancy", 0.0)
        if occ > max_occ:
            max_occ = occ
            best_warp = r.get("num_warps")
    return {"best_num_warps": best_warp, "max_occupancy": max_occ}
