import os
import torch


def verify_fallback_toggle(model, sample_input):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dev = model.to(device)
    x = sample_input.to(device)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    enabled_success = False
    try:
        with torch.no_grad():
            _ = model_dev(x)
        enabled_success = True
    except Exception:
        enabled_success = False
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    disabled_crashed = False
    try:
        with torch.no_grad():
            _ = model_dev(x)
        disabled_crashed = False
    except Exception:
        disabled_crashed = True
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return {"enabled_success": enabled_success, "disabled_crashed": disabled_crashed}
