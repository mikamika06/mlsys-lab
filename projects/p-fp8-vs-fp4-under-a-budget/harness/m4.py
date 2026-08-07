def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from quant.selector import check_kernel_support

    m = {"kernel_support_ok": 0.0}
    s_hop = check_kernel_support("fp4", "hopper")
    s_bla = check_kernel_support("fp4", "blackwell")
    if s_hop is False and s_bla is True:
        m["kernel_support_ok"] = 1.0
    return m
