import torch

def check_equivalence(compiled_out, eager_out, tol=1e-5):
    if isinstance(compiled_out, (tuple, list)):
        return all(check_equivalence(c, e, tol) for c, e in zip(compiled_out, eager_out))
    diff = torch.max(torch.abs(compiled_out - eager_out)).item()
    return diff <= tol
