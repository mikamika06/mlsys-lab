import torch


def compute_pruning_ratio(model, pruned_model):
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    return float(1.0 - (pruned_params / orig_params))


def get_dependency_groups(model):
    groups = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            groups.append({"name": name, "out_features": module.out_features, "in_features": module.in_features})
    return groups
