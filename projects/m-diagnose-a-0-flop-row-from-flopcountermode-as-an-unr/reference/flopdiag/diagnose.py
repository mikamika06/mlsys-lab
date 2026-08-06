def diagnose_row(op_name, flops, registered_ops):
    if flops == 0 and op_name not in registered_ops:
        return "unregistered_custom_op"
    return "valid"
