import ref

def check(workdir):
    from triage.memory import bakeoff_kv_memory
    from triage.matrix import extract_support_matrix

    out = {"memory_match": 0.0, "matrix_match": 0.0}

    cfg_a, cfg_b, target = ref.BAKEOFF_TESTS[0]
    try:
        mem_res = bakeoff_kv_memory(cfg_a, cfg_b, target)
        if isinstance(mem_res, dict) and "engine_a" in mem_res and "engine_b" in mem_res:
            if mem_res["engine_a"]["num_blocks"] == mem_res["engine_b"]["num_blocks"]:
                out["memory_match"] = 1.0
    except Exception as e:
        out["_note"] = f"memory error: {str(e)[:100]}"

    src, expected_mat = ref.MATRIX_TESTS[0]
    try:
        mat_res = extract_support_matrix(src)
        if mat_res == expected_mat:
            out["matrix_match"] = 1.0
    except Exception as e:
        out["_note"] = f"matrix error: {str(e)[:100]}"

    return out
