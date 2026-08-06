import ref
import torch

def check(workdir):
    from export_util.fixup import fix_data_dependent_flow
    from export_util.export import export_with_dynamic_batch
    out = {"roundtrip_matched": 0.0, "max_error_ok": 0.0}
    try:
        model = ref.DummyModel()
        fixed = fix_data_dependent_flow(model)
        inputs = ref.get_sample_inputs()
        ep = export_with_dynamic_batch(fixed, inputs)
        res = ep.module()(*inputs)
        orig_res = model(*inputs)
        err = torch.max(torch.abs(res - orig_res)).item()
        if err < 1e-5:
            out["max_error_ok"] = 1.0
        out["roundtrip_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"m2 failed: {type(e).__name__}: {str(e)[:120]}"
    return out
