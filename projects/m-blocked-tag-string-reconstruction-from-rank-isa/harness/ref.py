CASES = [
    {"rank": 4, "isa": "avx512", "want": "Acdb16a"},
    {"rank": 4, "isa": "avx2", "want": "acdb8a"},
    {"rank": 2, "isa": "avx512", "want": "ab"},
    {"rank": 3, "isa": "sse42", "want": "abc"},
]

LOGS = [
    ("onednn_verbose,info,reorder,src_nchw,dst_nhwc\nonednn_verbose,info,reorder,src_nchw,dst_nchw", {"nchw": 2, "nhwc": 1}),
    ("no reorders here", {"nchw": 0, "nhwc": 0}),
]
