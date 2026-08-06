import ref


def check(workdir):
    from triton_prof.interpret import build_block_table
    out = {"table_matched": 0.0}
    grid = (4,)
    durations = [15.0, 14.5, 16.2, 15.8]
    want = ref.build_block_table(grid, durations)
    try:
        got = build_block_table(grid, durations)
    except Exception:
        got = None
    if got == want:
        out["table_matched"] = 1.0
    return out
