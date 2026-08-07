import torch


def fine_tune_recovery(model: torch.nn.Module, dataloader, epochs: int, lr: float) -> torch.nn.Module:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    model.eval()
    return model
