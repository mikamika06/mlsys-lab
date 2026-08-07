import torch

def run_scaling_loop(model, optimizer, data_stream, inject_inf_steps=None):
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    scales = []
    for step, (x, y) in enumerate(data_stream):
        optimizer.zero_grad()
        with torch.amp.autocast("cpu", dtype=torch.float16):
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, y)
        scaler.scale(loss).backward()
        if inject_inf_steps and step in inject_inf_steps:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("inf"))
        scaler.step(optimizer)
        scaler.update()
        scales.append(scaler.get_scale())
    return scales
