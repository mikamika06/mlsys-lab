def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from lora_merge.merger import LoRAMerger
    import ref

    m = {"merge_safe": 0.0}
    base, a, b, _ = ref.get_test_data()
    merger = LoRAMerger(base, a, b, alpha=4.0, rank=2)
    merged = merger.safe_merge()
    if merged is not None and len(merged) == len(base):
        m["merge_safe"] = 1.0
    return m
