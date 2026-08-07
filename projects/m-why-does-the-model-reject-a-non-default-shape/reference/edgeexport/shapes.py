def validate_shape(shape_spec, actual_shape):
    if len(shape_spec) != len(actual_shape):
        return False
    for spec_dim, act_dim in zip(shape_spec, actual_shape):
        if isinstance(spec_dim, int):
            if spec_dim != act_dim:
                return False
        elif isinstance(spec_dim, dict):
            min_v = spec_dim.get("min", 1)
            max_v = spec_dim.get("max", 65536)
            if not (min_v <= act_dim <= max_v):
                return False
    return True


def evaluate_enumeration(shapes, constraints):
    valid = []
    for s in shapes:
        ok = True
        for c in constraints:
            dim_idx = c["dim"]
            val = s[dim_idx]
            if "min" in c and val < c["min"]:
                ok = False
            if "max" in c and val > c["max"]:
                ok = False
        if ok:
            valid.append(s)
    return valid
