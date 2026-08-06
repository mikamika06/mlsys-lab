def run_dummy_finetune(model, optimizer, data):
    model.train()
    init_loss = None
    final_loss = None
    for i, batch in enumerate(data):
        optimizer.zero_grad()
        out = model(batch)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        if i == 0:
            init_loss = loss.item()
        final_loss = loss.item()
    return {"loss_decreased": final_loss < init_loss}
