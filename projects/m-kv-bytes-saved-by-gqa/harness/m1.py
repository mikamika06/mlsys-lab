import ref


def check(workdir):
    try:
        from gqa_opt.classifier import classify_attention, compute_group_mapping
    except Exception as e:
        return {"configs_correct": 0.0, "_note": f"Import error: {e}"}

    total = len(ref.CONFIGS)
    matched = 0

    for i, cfg in enumerate(ref.CONFIGS):
        ref_cls = ref.classify_attention(cfg)
        ref_map = ref.compute_group_mapping(cfg)

        try:
            got_cls = classify_attention(cfg)
            got_map = compute_group_mapping(cfg)
        except Exception as e:
            return {
                "configs_correct": 0.0,
                "_note": f"Config {i} raised exception: {e}",
            }

        if got_cls == ref_cls and got_map == ref_map:
            matched += 1
        else:
            return {
                "configs_correct": 0.0,
                "_note": f"Mismatch at config {i}: got cls={got_cls}, map={got_map}; want cls={ref_cls}, map={ref_map}",
            }

    return {"configs_correct": float(matched == total)}
