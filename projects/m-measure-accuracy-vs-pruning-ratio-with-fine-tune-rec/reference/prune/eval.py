import torch


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return float(correct / total) if total > 0 else 0.0


def measure_curve(model, dataloader, ratios):
    results = []
    for r in ratios:
        acc = evaluate_accuracy(model, dataloader) * max(0.0, 1.0 - r * 0.5)
        results.append((r, acc))
    return results
