import ref
import torch

def check(workdir):
    from fusedmem.triton_grad import verify_gradients
    matched = 0
    for x in ref.TEST_INPUTS:
        x_naive = x.clone().detach().requires_grad_(True)
        x_fused = x.clone().detach().requires_grad_(True)
        if verify_gradients(x_naive, x_fused):
            matched += 1
    return {"grad_matched": 1.0 if matched == len(ref.TEST_INPUTS) else 0.0}
