import ref


def check(workdir):
    from profiler.interpreter import build_block_timing_table
    out = {"table_matched": 0.0}
    ok = True
    for i, t in enumerate(ref.GRID_TESTS):
        want = ref.build_block_timing_table(t["grid"], t["base_time"])
        got = build_block_timing_table(t["grid"], t["base_time"])
        if got is None or len(got) != len(want):
            ok = False
            out["_note"] = f"test {i}: length mismatch"
            break
        for g_row, w_row in zip(got, want):
            if g_row.get("block_id") != w_row["block_id"] or abs(g_row.get("duration_us", -1) - w_row["duration_us"]) > 1e-4:
                ok = False
                out["_note"] = f"test {i}: row mismatch got {g_row}, want {w_row}"
                break
        if not ok:
            break
    if ok:
        out["table_matched"] = 1.0
    return out
