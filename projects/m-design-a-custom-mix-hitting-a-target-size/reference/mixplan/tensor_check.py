def verify_1d_tensors(tensors, ftypes):
    for t in tensors:
        shape = t.get("shape", [])
        if len(shape) == 1:
            for ft in ftypes:
                assigned = t.get("ftype_map", {}).get(ft, "F32")
                if assigned != "F32":
                    return False
    return True
