import torch
import torch.nn as nn


class LegacyModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, mask=None):
        out = self.fc(x)
        if mask is not None:
            mask_sum = mask.sum().item()
            if mask_sum > 0:
                out = out * mask
        if out.mean().item() > 0.5:
            out = torch.relu(out)
        else:
            out = torch.sigmoid(out)
        return out


class CleanModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, mask=None):
        out = self.fc(x)
        if mask is not None:
            out = out * mask
        cond = out.mean() > 0.5
        out = torch.where(cond, torch.relu(out), torch.sigmoid(out))
        return out
