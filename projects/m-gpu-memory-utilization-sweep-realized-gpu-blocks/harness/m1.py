import ref

def check(workdir):
    from gpucache.sweep import compute_realized_blocks
    total = 80 * 1024**3
    reserved = 10 * 1024**3
    block_size = 65536
    utils = [0.5, 0.75, 0.9, 0.95]
    max_err = 0.0
    for u in utils:
        want = ref.compute_realized_blocks(total, reserved, u, block_size)
        try:
            got = compute_realized_blocks(total, reserved, u, block_size)
        except Exception:
            got = -1
        if want == 0:
            err = 0.0 if got == 0 else 1.0
        else:
            err = abs(got - want) / float(want)
        if err > max_err:
            max_err = err
    return {"rel_err": float(max_err)}
