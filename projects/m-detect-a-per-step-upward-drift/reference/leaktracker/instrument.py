import torch


def instrument_loop(model, dataloader, steps):
    torch.cuda.memory._record_memory_history(max_entries=100000)
    history = []
    iterator = iter(dataloader)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        outputs = model(batch)
        loss = outputs.sum()
        loss.backward()
        snapshot = torch.cuda.memory._snapshot()
        history.append(snapshot)
        torch.cuda.zero_grad_tensor_cache()
    torch.cuda.memory._record_memory_history(enabled=None)
    return history
