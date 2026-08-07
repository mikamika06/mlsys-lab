import torch

def train_stable_steps(model, dataloader, optimizer, scaler, steps=1000):
    model.train()
    completed = 0
    for batch in dataloader:
        if completed >= steps:
            break
        inputs, targets = batch
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        completed += 1
    return completed
