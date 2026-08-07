import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(64, 64)
        self.classifier = nn.Linear(64, 4)


def build_test_model(freeze_backbone=True, train_head=True):
    model = DummyModel()
    for param in model.backbone.parameters():
        param.requires_grad = not freeze_backbone
    for param in model.classifier.parameters():
        param.requires_grad = train_head
    return model


def build_adapter_dict():
    return {
        "classifier.weight": torch.randn(4, 64),
        "classifier.bias": torch.randn(4),
        "lora_a": torch.randn(8, 64),
        "lora_b": torch.randn(64, 8),
    }
