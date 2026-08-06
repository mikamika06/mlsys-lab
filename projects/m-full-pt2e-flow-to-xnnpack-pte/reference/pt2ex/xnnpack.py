import torch


def export_to_pte(model, sample_inputs, path):
    model.eval()
    with torch.no_grad():
        if isinstance(sample_inputs, (list, tuple)):
            traced = torch.jit.trace(model, sample_inputs)
        else:
            traced = torch.jit.trace(model, (sample_inputs,))
    torch.jit.save(traced, path)
    return path


def evaluate_error(ref_outputs, quantized_outputs):
    if isinstance(ref_outputs, (list, tuple)):
        errs = [torch.max(torch.abs(r - q)).item() for r, q in zip(ref_outputs, quantized_outputs)]
        return max(errs)
    return torch.max(torch.abs(ref_outputs - quantized_outputs)).item()
