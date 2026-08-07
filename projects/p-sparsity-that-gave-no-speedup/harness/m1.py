def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import numpy as np
    import ref
    from sparse_eval.pattern import compress_24_matrix, decompress_24_matrix, verify_24_pattern

    res = {
        "valid_pattern_detected": 0.0,
        "invalid_pattern_rejected": 0.0,
        "compression_shape_ok": 0.0,
        "decompression_exact": 0.0,
    }

    sp_mat = ref.generate_24_sparse_matrix(32, 64)
    dn_mat = ref.generate_dense_matrix(32, 64)

    if verify_24_pattern(sp_mat):
        res["valid_pattern_detected"] = 1.0

    if not verify_24_pattern(dn_mat):
        res["invalid_pattern_rejected"] = 1.0

    cw, meta = compress_24_matrix(sp_mat)
    if cw.shape == (32, 32) and meta.shape == (32, 16):
        res["compression_shape_ok"] = 1.0

    rec = decompress_24_matrix(cw, meta, 64)
    if float(np.max(np.abs(sp_mat - rec))) < 1e-5:
        res["decompression_exact"] = 1.0

    return res
