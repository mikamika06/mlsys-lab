import torch
import torch.nn as nn
import torch_pruning as tp


def prune_real_cnn(model, example_inputs, size_ratio):
    strategy = tp.strategy.L1Strategy()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=strategy,
        size_ratio=size_ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return model
