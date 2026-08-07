import torch

def analyze_scaler_step(scaler, optimizer, loss):
    optimizer.zero_grad()
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()

    found_inf = False
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                found_inf = True
                break

    initial_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    new_scale = scaler.get_scale()

    skipped = new_scale < initial_scale or (found_inf and new_scale <= initial_scale)
    return {"scale": initial_scale, "new_scale": new_scale, "skipped": skipped, "inf_detected": found_inf}
