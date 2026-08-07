import ref

def check(workdir):
    from zeromem.reducescatter import toy_reduce_scatter
    out = {"reduce_scatter_matched": 0.0}
    grads = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    ws = 4
    matched = True
    for rank in range(ws):
        want = ref.toy_reduce_scatter(grads, ws, rank)
        try:
            got = toy_reduce_scatter(grads, ws, rank)
        except Exception as e:
            matched = False
            out["_note"] = f"raised exception: {e}"
            break
        if list(got) != list(want):
            matched = False
            out["_note"] = f"rank {rank}: got {got}, want {want}"
            break
    if matched:
        out["reduce_scatter_matched"] = 1.0
    return out
