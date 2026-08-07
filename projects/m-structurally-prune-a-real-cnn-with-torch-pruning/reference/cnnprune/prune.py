import time
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def structural_prune(model, prune_ratio):
    with torch.no_grad():
        w = model.conv1.weight
        out_channels = w.size(0)
        keep_count = int(out_channels * (1.0 - prune_ratio))
        indices = torch.randperm(out_channels)[:keep_count].sort().values
        model.conv1.weight = nn.Parameter(w[indices].clone())
        model.conv1.bias = nn.Parameter(model.conv1.bias[indices].clone())
        model.conv1.out_channels = keep_count

        w2 = model.conv2.weight
        model.conv2.weight = nn.Parameter(w2[:, indices].clone())
        model.conv2.in_channels = keep_count

        fc_w = model.fc.weight
        expected_fc_in = keep_count * 32 * 32
        model.fc.weight = nn.Parameter(fc_w[:, :expected_fc_in].clone())
        model.fc.in_features = expected_fc_in
    return model


def measure_speedup(model_orig, model_pruned, sample_input):
    model_orig.eval()
    model_pruned.eval()
    with torch.no_grad():
        for _ in range(10):
            model_orig(sample_input)
        t0 = time.time()
        for _ in range(50):
            model_orig(sample_input)
        t_orig = time.time() - t0

        for _ in range(10):
            model_pruned(sample_input)
        t0 = time.time()
        for _ in range(50):
            model_pruned(sample_input)
        t_pruned = time.time() - t0

    orig_size = sum(p.numel() * p.element_size() for p in model_orig.parameters())
    pruned_size = sum(p.numel() * p.element_size() for p in model_pruned.parameters())
    return float(pruned_size) / float(orig_size)
