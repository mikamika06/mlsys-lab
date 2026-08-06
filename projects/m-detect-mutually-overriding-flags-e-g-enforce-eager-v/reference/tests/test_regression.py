from vllmargs.clamp import find_clamping_arg


def test_clamping_detection():
    cfg = {"max_model_len": 1024, "max_num_seqs": 8}
    res = find_clamping_arg(100000, cfg)
    assert res is not None
