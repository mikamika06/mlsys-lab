def find_offending_program_id(grid_shape, test_func):
    for pid in range(grid_shape):
        try:
            test_func(pid)
        except Exception:
            return pid
    return -1
