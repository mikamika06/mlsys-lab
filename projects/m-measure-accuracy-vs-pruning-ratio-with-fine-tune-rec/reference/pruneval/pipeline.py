import copy
import torch
import torch_pruning as tp


def compute_pruned_model(model: torch.nn.Module, example_inputs: torch.Tensor, ratio: float) -> torch.nn.Module:
    if ratio <= 0.0:
        return model
    model_copy = copy.deepcopy(model)
    imp = tp.importance.GroupNormImportance(p=2)
    ignored_layers = []
    for m in model_copy.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner(
        model_copy,
        example_inputs,
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return model_copy
