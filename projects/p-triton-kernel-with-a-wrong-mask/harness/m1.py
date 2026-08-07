import sys


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"non_block_multiple_reproduced": 0.0, "block_multiple_ok": 0.0}
    try:
        import triton_mask.kernel as k

        out_non_mult = k.process_data(list(range(1, 101)), 100, BLOCK_SIZE=64)
        if len(out_non_mult) == 100:
            res["non_block_multiple_reproduced"] = 1.0

        out_mult = k.process_data(list(range(1, 129)), 128, BLOCK_SIZE=64)
        if len(out_mult) == 128:
            res["block_multiple_ok"] = 1.0
    except Exception:
        pass
    return res
