import torch

def compare_fp16_bf16(model_fn, data_stream, inject_inf_steps):
    m_fp16 = model_fn()
    opt_fp16 = torch.optim.SGD(m_fp16.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    scales_fp16 = []
    for step, (x, y) in enumerate(data_stream):
        opt_fp16.zero_grad()
        with torch.amp.autocast("cpu", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(m_fp16(x), y)
        scaler.scale(loss).backward()
        if inject_inf_steps and step in inject_inf_steps:
            for p in m_fp16.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("inf"))
        scaler.step(opt_fp16)
        scaler.update()
        scales_fp16.append(scaler.get_scale())
    m_bf16 = model_fn()
    opt_bf16 = torch.optim.SGD(m_bf16.parameters(), lr=0.01)
    loss_history_bf16 = []
    for step, (x, y) in enumerate(data_stream):
        opt_bf16.zero_grad()
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            loss = torch.nn.functional.mse_loss(m_bf16(x), y)
        loss.backward()
        if inject_inf_steps and step in inject_inf_steps:
            for p in m_bf16.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("inf"))
        opt_bf16.step()
        loss_history_bf16.append(loss.item())
    return {"scales_fp16": scales_fp16, "loss_bf16": loss_history_bf16}
