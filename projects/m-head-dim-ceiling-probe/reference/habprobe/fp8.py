def check_fp8_availability(head_dim, precision):
    if precision != "fp8":
        return True
    return head_dim == 128 or head_dim == 64
