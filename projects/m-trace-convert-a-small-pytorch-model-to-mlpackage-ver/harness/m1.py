import os
import torch
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    from coreml_exporter.converter import export_and_verify

    out = {"models_converted": 0.0, "max_abs_err": 1.0}

    model = ref.SampleModel()
    example_inputs, eval_inputs = ref.get_sample_inputs()
    save_path = os.path.join(workdir, "test_model.mlpackage")

    try:
        mlmodel, max_err = export_and_verify(model, example_inputs, eval_inputs, save_path)
        out["models_converted"] = 1.0
        out["max_abs_err"] = float(max_err)
    except Exception as e:
        out["_note"] = f"export_and_verify failed: {type(e).__name__}: {str(e)}"

    return out
