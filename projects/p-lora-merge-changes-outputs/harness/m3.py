def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from lora_merge.merger import LoRAMerger
    import ref

    m = {"scaling_correct": 0.0}
    base, a, b, _ = ref.get_test_data()
    merger = LoRAMerger(base, a, b, alpha=4.0, rank=2)
    if merger.verify_scaling():
        m["scaling_correct"] = 1.0
    return m
