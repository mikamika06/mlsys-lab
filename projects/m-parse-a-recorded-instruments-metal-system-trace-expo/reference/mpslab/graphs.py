def compare_execution(mode, op_count):
    if mode == "loop":
        return op_count
    elif mode == "graph":
        return 1
    raise ValueError("unknown mode")
