import ref

def check(workdir):
    from zigzag.comm import compute_comm_volume
    from zigzag.overlap import check_overlap_feasibility

    out = {"volume_match": 0.0, "feasibility_match": 0.0}

    num_tokens, world_size, head_dim, dtype_size = 128, 4, 64, 2
    want_vol = ref.compute_comm_volume(num_tokens, world_size, head_dim, dtype_size)
    got_vol = compute_comm_volume(num_tokens, world_size, head_dim, dtype_size)
    if got_vol == want_vol:
        out["volume_match"] = 1.0

    f1 = check_overlap_feasibility(20.0, 15.0, 0.5)
    f2 = check_overlap_feasibility(50.0, 10.0, 0.1)
    wf1 = ref.check_overlap_feasibility(20.0, 15.0, 0.5)
    wf2 = ref.check_overlap_feasibility(50.0, 10.0, 0.1)
    if f1 == wf1 and f2 == wf2:
        out["feasibility_match"] = 1.0

    return out
