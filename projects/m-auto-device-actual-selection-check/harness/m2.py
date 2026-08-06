import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    try:
        from ovdev.matrix import build_compile_matrix
        import ref as harness_ref
    except Exception as e:
        return {"matrix_matched": 0.0, "_note": f"Import error: {e}"}

    devices, shapes = harness_ref.generate_matrix_specs()

    ref_matrix = {}
    for dev in devices:
        dev_name = dev["name"]
        ref_matrix[dev_name] = {}
        for shape in shapes:
            s_name = shape["name"]
            is_dyn = shape["is_dynamic"]
            rank = len(shape["dims"])
            
            if rank > dev["max_dims"]:
                succ = False
                reason = f"Rank {rank} exceeds device max {dev['max_dims']}"
            elif is_dyn and not dev["supports_dynamic"]:
                succ = False
                reason = f"Device {dev_name} does not support dynamic shapes"
            elif not is_dyn and any(d <= 0 for d in shape["dims"]):
                succ = False
                reason = "Invalid non-positive static dimension"
            else:
                succ = True
                reason = "OK"

            ref_matrix[dev_name][s_name] = {"success": succ, "reason": reason}

    try:
        got_matrix = build_compile_matrix(devices, shapes)
    except Exception as e:
        return {"matrix_matched": 0.0, "_note": f"Execution error: {e}"}

    if got_matrix == ref_matrix:
        return {"matrix_matched": 1.0}

    return {
        "matrix_matched": 0.0,
        "_note": f"Matrix mismatch. Reference keys: {list(ref_matrix.keys())}, Got keys: {list(got_matrix.keys()) if isinstance(got_matrix, dict) else type(got_matrix)}"
    }
