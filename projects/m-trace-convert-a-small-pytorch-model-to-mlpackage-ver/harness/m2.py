import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    from coreml_exporter.precision import compare_precisions

    out = {"precision_measured": 0.0, "size_ratio_fp16_fp32": 1.0}

    model = ref.SampleModel()
    example_inputs, eval_inputs = ref.get_sample_inputs()

    try:
        res = compare_precisions(model, example_inputs, eval_inputs, workdir)
        if isinstance(res, dict) and "ratio" in res:
            out["precision_measured"] = 1.0
            out["size_ratio_fp16_fp32"] = float(res["ratio"])
        else:
            out["_note"] = "compare_precisions did not return a valid metrics dict"
    except Exception as e:
        out["_note"] = f"compare_precisions failed: {type(e).__name__}: {str(e)}"

    return out
