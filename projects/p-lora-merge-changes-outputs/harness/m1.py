def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    from lora_merge.merger import LoRAMerger
    import ref

    m = {"layers_measured": 0.0}
    base, a, b, prompts = ref.get_test_data()
    merger = LoRAMerger(base, a, b, alpha=4.0, rank=2)
    diffs = merger.measure_layer_diffs(prompts[0])
    if isinstance(diffs, list) and len(diffs) == len(base):
        m["layers_measured"] = 1.0
    return m
