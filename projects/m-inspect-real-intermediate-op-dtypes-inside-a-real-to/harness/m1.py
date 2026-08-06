import ref
import torch
import sys

def check(workdir):
    out = {"output_matched": 0.0, "activations_matched": 0.0, "weights_matched": 0.0}
    sys.path.insert(0, workdir)
    try:
        from autocast_inspect.inspector import inspect_autocast
    except ImportError:
        out["_note"] = "Could not import inspect_autocast"
        return out

    model = ref.get_model()
    x = ref.get_input()

    try:
        want = ref.check_m1_oracle(model, x)
        got = inspect_autocast(model, x, "cpu", torch.bfloat16)
    except Exception as e:
        out["_note"] = f"Error running inspect_autocast: {e}"
        return out

    if got.get("output_dtype") == want["output_dtype"]:
        out["output_matched"] = 1.0
    if got.get("activation_dtypes") == want["activation_dtypes"]:
        out["activations_matched"] = 1.0
    if got.get("weight_dtypes") == want["weight_dtypes"]:
        out["weights_matched"] = 1.0

    if out["output_matched"] == 0.0:
        out["_note"] = f"output_dtype mismatch: got {got.get('output_dtype')}, want {want['output_dtype']}"
    elif out["activations_matched"] == 0.0:
        out["_note"] = f"activation_dtypes mismatch: got {got.get('activation_dtypes')}, want {want['activation_dtypes']}"
    elif out["weights_matched"] == 0.0:
        out["_note"] = f"weight_dtypes mismatch: got {got.get('weight_dtypes')}, want {want['weight_dtypes']}"

    return out
