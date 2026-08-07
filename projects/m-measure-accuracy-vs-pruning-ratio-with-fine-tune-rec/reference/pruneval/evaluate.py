import torch
from pruneval.pipeline import compute_pruned_model
from pruneval.recovery import fine_tune_recovery


def evaluate_accuracy(model: torch.nn.Module, dataloader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return float(correct) / float(max(1, total))


def measure_curve(model: torch.nn.Module, dataloader, example_inputs: torch.Tensor, ratios: list) -> dict:
    results = {}
    for r in ratios:
        pruned = compute_pruned_model(model, example_inputs, r)
        recovered = fine_tune_recovery(pruned, dataloader, epochs=1, lr=0.01)
        acc = evaluate_accuracy(recovered, dataloader)
        results[r] = acc
    return results
