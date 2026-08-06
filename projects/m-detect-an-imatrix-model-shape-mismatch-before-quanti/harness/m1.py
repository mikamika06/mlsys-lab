import ref

def check(workdir):
    from imatrix.core import verify_tensor_shape
    tg, ig, tb, ib = ref.get_test_cases()
    ok_count = 0
    total = 2
    try:
        ok1, _ = verify_tensor_shape("blk.0.attn_q.weight", tg["blk.0.attn_q.weight"], ig["blk.0.attn_q.weight"])
        if ok1:
            ok_count += 1
    except Exception:
        pass
    try:
        ok2, _ = verify_tensor_shape("blk.0.attn_q.weight", tb["blk.0.attn_q.weight"], ib["blk.0.attn_q.weight"])
        if not ok2:
            ok_count += 1
    except Exception:
        pass
    return {"tensor_checks_passed": float(ok_count == total)}
