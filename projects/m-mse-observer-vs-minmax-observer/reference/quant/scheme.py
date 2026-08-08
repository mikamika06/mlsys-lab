def parse_scheme(scheme_name):
    parts = scheme_name.split("-")
    dtype = parts[0].lower()
    symmetric = parts[1].lower() == "sym"
    granularity = parts[2].lower()

    if dtype == "int8":
        qmin, qmax = -128, 127
    elif dtype == "uint8":
        qmin, qmax = 0, 255
    elif dtype == "int4":
        qmin, qmax = -8, 7
    elif dtype == "uint4":
        qmin, qmax = 0, 15
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return {
        "dtype": dtype,
        "symmetric": symmetric,
        "granularity": granularity,
        "qmin": qmin,
        "qmax": qmax
    }
