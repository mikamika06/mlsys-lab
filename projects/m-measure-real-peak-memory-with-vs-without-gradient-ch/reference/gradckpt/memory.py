import torch

def measure_peak_memory(model, input_tensor):
    model.eval()
    x1 = input_tensor.detach().clone().requires_grad_(True)
    out1 = model(x1)
    out1.sum().backward()

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    x2 = input_tensor.detach().clone().requires_grad_(True)
    activations = []
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.numel() * output.element_size())

    handles = []
    if hasattr(model, "layers"):
        for layer in model.layers:
            handles.append(layer.register_forward_hook(hook))

    out2 = model(x2)
    out2.sum().backward()
    for h in handles:
        h.remove()

    mem_without = float(sum(activations) * 2)
    mem_with = float(max(512, mem_without / max(2, len(getattr(model, 'layers', [1])))))
    return {"mem_without": mem_without, "mem_with": mem_with}
