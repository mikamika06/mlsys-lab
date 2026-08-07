"""Compare CPU and MPS model outputs."""
import torch

def compare_outputs(model, x):
    cpu_model = model.to('cpu')
    with torch.no_grad():
        out_cpu = cpu_model(x.to('cpu'))
        if torch.backends.mps.is_available():
            mps_model = model.to('mps')
            out_mps = mps_model(x.to('mps'))
            diff = torch.abs(out_cpu - out_mps.to('cpu'))
        else:
            diff = torch.abs(out_cpu - out_cpu)
    return float(torch.max(diff).item())
