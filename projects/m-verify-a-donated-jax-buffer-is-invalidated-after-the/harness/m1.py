import ref

def check(workdir):
    out = {"donation_detected": 0.0, "deleted_buffers_identified": 0.0}
    try:
        from jaxserv.donation import is_buffer_deleted, verify_donation
    except Exception as e:
        out["_note"] = f"Failed to import donation module: {type(e).__name__}: {e}"
        return out

    buf_deleted = ref.MockJaxBuffer([1, 2, 3], is_donated=True)
    buf_kept = ref.MockJaxBuffer([4, 5, 6], is_donated=False)

    res_d, inv_d = verify_donation(ref.mock_jit_donate_call, buf_deleted)
    res_k, inv_k = verify_donation(ref.mock_jit_donate_call, buf_kept)

    if res_d == [2, 3, 4] and inv_d is True and inv_k is False:
        out["donation_detected"] = 1.0

    dummy_buf = ref.MockJaxBuffer([], is_donated=True)
    dummy_buf._deleted = True
    if is_buffer_deleted(dummy_buf) is True and is_buffer_deleted(buf_kept) is False:
        out["deleted_buffers_identified"] = 1.0

    return out
