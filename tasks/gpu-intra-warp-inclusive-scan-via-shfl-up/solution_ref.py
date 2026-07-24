def inclusive_scan_warp(t, n):
    """Hillis-Steele scan inside one warp, using shuffles only."""
    i = t.gid
    if i >= n:
        return
    v = t.gload(i)
    for delta in (1, 2, 4, 8, 16):
        up = yield t.shfl_up(v, delta)
        if t.lane >= delta:
            v += up
            t.alu(1)
    t.gstore(i, v)
