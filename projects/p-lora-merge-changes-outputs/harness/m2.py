def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from lora_merge.merger import LoRAMerger
    import ref

    m = {"dtype_fixed": 0.0}
    base, a, b, _ = ref.get_test_data()
    merger = LoRAMerger(base, a, b, alpha=4.0, rank=2)
    res = merger.fix_dtype()
    if res is True:
        m["dtype_fixed"] = 1.0
    return m
