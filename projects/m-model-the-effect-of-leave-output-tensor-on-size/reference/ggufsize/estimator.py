from ggufsize.calc import model_total_bytes


def estimate_difference(tensors, ftype):
    with_out = model_total_bytes(tensors, ftype, leave_output=True)
    without_out = model_total_bytes(tensors, ftype, leave_output=False)
    return with_out - without_out
