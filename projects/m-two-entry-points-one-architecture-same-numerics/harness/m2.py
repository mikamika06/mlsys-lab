import ref

def check(workdir):
    from tflite_pipe.opdiff import get_op_diff
    out = {"diff_matched": 0.0}
    ops1 = ["ADD", "MUL", "SELECT_TF_OPS:Slice"]
    ops2 = ["ADD", "MUL", "RESIZE_BILINEAR"]
    want = ref.compute_op_diff(ops1, ops2)
    got = get_op_diff(ops1, ops2)
    if got is not None and sorted(got) == sorted(want):
        out["diff_matched"] = 1.0
    return out
