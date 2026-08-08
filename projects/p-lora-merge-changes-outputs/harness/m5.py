def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from lora_merge.merger import LoRAMerger
    import ref

    m = {"error_below_threshold": 0.0}
    base, a, b, prompts = ref.get_test_data()
    merger = LoRAMerger(base, a, b, alpha=4.0, rank=2)
    merger.safe_merge()
    err = merger.evaluate_prompts(prompts)
    if err < 1e-4:
        m["error_below_threshold"] = 1.0
    return m
