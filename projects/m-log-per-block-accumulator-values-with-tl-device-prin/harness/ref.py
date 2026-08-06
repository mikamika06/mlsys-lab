def generate_print_logs():
    return [
        ("[0, 0, 0] acc_val: 10.0\n[1, 0, 0] acc_val: 25.5\n", {(0,0,0): 10.0, (1,0,0): 25.5}),
        ("[0, 1, 0] acc_val: -1.0\n", {(0,1,0): -1.0}),
        ("[12, 34, 56] acc_val: 100.0\n", {(12,34,56): 100.0}),
        ("junk line\n[0, 0, 0] other_val: 5.5", {(0,0,0): 5.5})
    ]

def generate_error_logs():
    return [
        ("Traceback...\nValueError: out-of-bounds memory access at program_id (2, 1, 0)", (2, 1, 0)),
        ("RuntimeError: some error at program_id (10, 0, 5)", (10, 0, 5)),
        ("Exception: mask missing at program_id (123, 456, 789) - aborting", (123, 456, 789)),
        ("No program id here", ())
    ]
