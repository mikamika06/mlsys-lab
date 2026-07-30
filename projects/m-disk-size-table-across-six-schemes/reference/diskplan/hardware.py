from diskplan.schemes import disk_size


def hardware_native(scheme, hardware):
    return scheme["bits"] in hardware["native_bits"]


def gate_table(model, hardware, schemes):
    baseline = disk_size(model, schemes[0])
    rows = []
    for scheme in schemes:
        b = disk_size(model, scheme)
        rows.append({
            "scheme": scheme["name"],
            "bits": scheme["bits"],
            "bytes": b,
            "ratio": b / baseline,
            "native": hardware_native(scheme, hardware),
        })
    return rows


def best_native_scheme(model, hardware, schemes):
    best_name = None
    best_bytes = None
    for scheme in schemes:
        if not hardware_native(scheme, hardware):
            continue
        b = disk_size(model, scheme)
        if best_bytes is None or b < best_bytes:
            best_name = scheme["name"]
            best_bytes = b
    return best_name
