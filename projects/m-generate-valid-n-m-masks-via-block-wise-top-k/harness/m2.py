import ref


def check(workdir):
    from sparse.error import capture_sparse_matmul_error
    out = {"error_captured": 0.0}
    try:
        err_type, err_msg = capture_sparse_matmul_error()
        if err_type is not None or isinstance(err_msg, str):
            out["error_captured"] = 1.0
        else:
            out["_note"] = "did not capture any error or exception info"
    except Exception as e:
        out["error_captured"] = 1.0
    return out
