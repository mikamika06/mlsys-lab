import torch

def check_autotune_mode(model, x):
    """Checks behavior of max-autotune mode on MPS."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = model.to(device)
    inp = x.to(device)

    status = "unknown"
    error_raised = False
    try:
        torch.compile(m, mode="max-autotune")
        status = "noop_or_success"
    except Exception:
        error_raised = True
        status = "error_raised"

    return {"status": status, "error_raised": error_raised}
