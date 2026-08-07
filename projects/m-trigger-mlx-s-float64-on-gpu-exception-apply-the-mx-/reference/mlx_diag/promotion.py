import mlx.core as mx


def compute_promotion_table(dtypes_a, dtypes_b):
    table = {}
    for da in dtypes_a:
        for db in dtypes_b:
            a = mx.array([1], dtype=da)
            b = mx.array([1], dtype=db)
            res = a + b
            table[(str(da), str(db))] = str(res.dtype)
    return table
