import ref
import torch


def check(workdir):
    from gcapture.dynshape import derive_minimal_dynamic_shapes

    out = {"spec_matched": 0.0}
    mod, example_input, failing_inputs = ref.get_dynshape_test_setup()

    got_spec = derive_minimal_dynamic_shapes(mod, example_input, failing_inputs)

    try:
        ep = torch.export.export(mod, example_input, dynamic_shapes=got_spec)
        for fail_inp in failing_inputs:
            ep.module()(*fail_inp)
        out["spec_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Export failed with generated spec: {type(e).__name__}: {e}"

    return out
