def compute_broadcast_shape(shape_a, shape_b):
    sa = list(shape_a)
    sb = list(shape_b)
    max_rank = max(len(sa), len(sb))
    pa = [1] * (max_rank - len(sa)) + sa
    pb = [1] * (max_rank - len(sb)) + sb
    out = []
    for da, db in zip(pa, pb):
        if da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        elif da == db:
            out.append(da)
        elif isinstance(da, str) and isinstance(db, int):
            out.append(da)
        elif isinstance(db, str) and isinstance(da, int):
            out.append(db)
        else:
            raise ValueError(f"Incompatible dimensions for broadcast: {da} and {db}")
    return out
