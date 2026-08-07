_FLOAT_RANKS = {"float64": 4, "float32": 3, "float16": 2, "bfloat16": 2}
_INT_RANKS = {"int64": 2, "int32": 1}

def _norm_dtype(dt):
    s = str(dt).lower().replace("dtype('", "").replace("')", "").strip()
    if s in ("float", "float_"):
        return "float64"
    if s in ("int", "int_"):
        return "int64"
    return s

def promote_dtypes(dt1, dt2):
    d1 = _norm_dtype(dt1)
    d2 = _norm_dtype(dt2)

    if d1 == d2:
        return d1
    if d1 == "bool":
        return d2
    if d2 == "bool":
        return d1

    is_f1 = d1 in _FLOAT_RANKS
    is_f2 = d2 in _FLOAT_RANKS
    is_i1 = d1 in _INT_RANKS
    is_i2 = d2 in _INT_RANKS

    if is_f1 and is_f2:
        if {d1, d2} == {"float16", "bfloat16"}:
            return "float32"
        r1 = _FLOAT_RANKS[d1]
        r2 = _FLOAT_RANKS[d2]
        return d1 if r1 >= r2 else d2

    if is_f1 and is_i2:
        return d1
    if is_f2 and is_i1:
        return d2

    if is_i1 and is_i2:
        r1 = _INT_RANKS[d1]
        r2 = _INT_RANKS[d2]
        return d1 if r1 >= r2 else d2

    return d1

def compute_promotion_table(dtypes):
    table = {}
    for d1 in dtypes:
        for d2 in dtypes:
            table[(str(d1), str(d2))] = promote_dtypes(d1, d2)
    return table
