import ref


def check(workdir):
    from zeroproj.partition import assign_partitions

    sizes = [100, 200, 300, 400, 500, 600]
    world_size = 3
    want = assign_partitions(sizes, world_size)
    try:
        got = assign_partitions(sizes, world_size)
    except Exception:
        got = None

    ok = 1 if got == want and got is not None else 0
    return {"partitions_matched": float(ok)}
