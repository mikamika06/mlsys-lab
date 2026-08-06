from dnnfmt.tags import reconstruct_tag
from dnnfmt.logs import count_reorders

def test_tag_reconstruction():
    assert reconstruct_tag(4, "avx512") == "Acdb16a"

def test_log_counting():
    log = "onednn_verbose,info,reorder,src_nchw,dst_nhwc"
    res = count_reorders(log)
    assert res["nchw"] == 1
    assert res["nhwc"] == 1
