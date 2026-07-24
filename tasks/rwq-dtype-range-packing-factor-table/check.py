def _expected_info(bit_width):
    q_min = - (1 << (bit_width - 1))
    q_max = (1 << (bit_width - 1)) - 1
    pack_factor = 8 // bit_width
    return (q_min, q_max, pack_factor)

def grade(sol, fx) -> dict:
    try:
        got = sol.dtype_range_packing()
    except Exception as e:
        return {"exact_match": 0.0}
    expected = {
        "qint2": _expected_info(2),
        "qint4": _expected_info(4),
        "qint8": _expected_info(8)
    }
    ok = 1.0 if got == expected else 0.0
    return {"exact_match": ok}
