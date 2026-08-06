def check_contiguity(strides):
    if not strides:
        return False
    return strides[-1] == 1

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

def generate_cases():
    base_q = {"shape": (2, 128, 8, 64), "strides": (65536, 512, 64, 1), "dtype": "float16"}
    base_k = {"shape": (2, 128, 2, 64), "strides": (16384, 128, 64, 1), "dtype": "float16"}
    base_v = {"shape": (2, 128, 2, 64), "strides": (16384, 128, 64, 1), "dtype": "float16"}

    cases = []
    cases.append((base_q, base_k, base_v))
    cases.append((dict(base_q, dtype="float32"), base_k, base_v))
    cases.append((dict(base_q, shape=(2, 128, 8)), base_k, base_v))
    cases.append((dict(base_q, strides=(65536, 512, 64, 2)), base_k, base_v))

    q4 = dict(base_q, shape=(2, 128, 8, 512))
    k4 = dict(base_k, shape=(2, 128, 2, 512))
    v4 = dict(base_v, shape=(2, 128, 2, 512))
    cases.append((q4, k4, v4))

    cases.append((dict(base_q, shape=(3, 128, 8, 64)), base_k, base_v))
    cases.append((dict(base_q, dtype="bfloat16"), dict(base_k, dtype="float16"), dict(base_v, dtype="float16")))

    return cases

CASES = generate_cases()

STRIDES_TO_TEST = [
    (65536, 512, 64, 1),
    (65536, 512, 64, 2),
    (1024, 1),
    (1024, 8),
    (),
    (1,)
]
