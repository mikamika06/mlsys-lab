import ref

def check(workdir):
    from fusion.bytes_calc import calc_bytes

    out = {"unfused_match": 0.0, "fused_match": 0.0}

    ops1 = [
        {"op": "add", "inputs": ["a", "b"], "output": "c"},
        {"op": "mul", "inputs": ["c", "d"], "output": "e"}
    ]
    ops2 = [
        {"op": "sub", "inputs": ["a", "b"], "output": "c"},
        {"op": "div", "inputs": ["c", "b"], "output": "d"},
        {"op": "exp", "inputs": ["d"], "output": "e"}
    ]

    try:
        g_u1 = calc_bytes(ops1, 2, 100, False)
        g_u2 = calc_bytes(ops2, 4, 50, False)
        w_u1 = ref.calc_bytes(ops1, 2, 100, False)
        w_u2 = ref.calc_bytes(ops2, 4, 50, False)
        if g_u1 == w_u1 and g_u2 == w_u2:
            out["unfused_match"] = 1.0

        g_f1 = calc_bytes(ops1, 2, 100, True)
        g_f2 = calc_bytes(ops2, 4, 50, True)
        w_f1 = ref.calc_bytes(ops1, 2, 100, True)
        w_f2 = ref.calc_bytes(ops2, 4, 50, True)
        if g_f1 == w_f1 and g_f2 == w_f2:
            out["fused_match"] = 1.0

    except Exception as e:
        out["_note"] = f"Failed calculating bytes: {e}"

    return out
