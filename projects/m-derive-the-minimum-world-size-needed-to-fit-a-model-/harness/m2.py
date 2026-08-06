import ref

def check(workdir):
    from fsdpfit.launcher import launch_and_get_sharded_sizes
    out = {"sharded_sizes_matched": 0.0}
    try:
        want = ref.launch_and_get_sharded_sizes()
        got = launch_and_get_sharded_sizes()
        if got == want:
            out["sharded_sizes_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = str(e)[:120]
    return out
