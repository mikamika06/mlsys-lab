from flash_contract.guard import check_contiguity

def classify_inputs(q, k, v):
    errors = []

    valid_dtypes = {"float16", "bfloat16"}
    if q["dtype"] not in valid_dtypes or k["dtype"] not in valid_dtypes or v["dtype"] not in valid_dtypes:
        errors.append("DTYPE_ERROR")
    elif not (q["dtype"] == k["dtype"] == v["dtype"]):
        errors.append("DTYPE_ERROR")

    if len(q["shape"]) != 4 or len(k["shape"]) != 4 or len(v["shape"]) != 4:
        errors.append("NDIM_ERROR")
        return sorted(errors)

    if not (check_contiguity(q["strides"]) and check_contiguity(k["strides"]) and check_contiguity(v["strides"])):
        errors.append("ALIGNMENT_ERROR")

    d_q, d_k, d_v = q["shape"][3], k["shape"][3], v["shape"][3]
    if not (d_q == d_k == d_v):
        errors.append("HEAD_DIM_ERROR")
    elif d_q > 256 or d_q % 8 != 0:
        errors.append("HEAD_DIM_ERROR")

    b_q, _, _, _ = q["shape"]
    b_k, s_k, h_k, _ = k["shape"]
    b_v, s_v, h_v, _ = v["shape"]

    if not (b_q == b_k == b_v):
        errors.append("SHAPE_ERROR")
    if s_k != s_v:
        errors.append("SHAPE_ERROR")
    if h_k != h_v:
        errors.append("SHAPE_ERROR")

    return sorted(errors)
