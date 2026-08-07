import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from quant.pipeline import run_quantization_pipeline

    out = {"size_reduction_ratio": 0.0, "accuracy_loss_within_floor": 0.0}

    model = ref.MockModel()
    dataset = ref.get_dataset()
    calib = ref.get_calib_data()

    try:
        res = run_quantization_pipeline(model, dataset, calib)
    except Exception:
        return out

    out["size_reduction_ratio"] = float(res.get("compression_ratio", 0.0))

    if res.get("accuracy_drop", 1.0) <= 0.05:
        out["accuracy_loss_within_floor"] = 1.0

    return out
