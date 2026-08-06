import ref
import torch

def check(workdir):
    from export_util.export import export_with_dynamic_batch
    out = {"dynamic_export_matched": 0.0}
    try:
        model = ref.DummyModel()
        inputs = ref.get_sample_inputs()
        ep = export_with_dynamic_batch(model, inputs)
        if ep is not None:
            out["dynamic_export_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"m1 failed: {type(e).__name__}: {str(e)[:120]}"
    return out
