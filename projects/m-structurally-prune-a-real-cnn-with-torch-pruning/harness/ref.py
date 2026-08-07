import torch
from cnnprune.toy import propagate_channels as ref_propagate


def get_expected_toy(pruned_channels):
    return ref_propagate(pruned_channels)


def structural_prune_ref(model, prune_ratio):
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        w = m.conv1.weight
        out_channels = w.size(0)
        keep_count = int(out_channels * (1.0 - prune_ratio))
        indices = torch.arange(keep_count)
        m.conv1.weight = torch.nn.Parameter(w[indices].clone())
        m.conv1.bias = torch.nn.Parameter(m.conv1.bias[indices].clone())
        m.conv1.out_channels = keep_count

        w2 = m.conv2.weight
        m.conv2.weight = torch.nn.Parameter(w2[:, indices].clone())
        m.conv2.in_channels = keep_count

        fc_w = m.fc.weight
        expected_fc_in = keep_count * 32 * 32
        m.fc.weight = torch.nn.Parameter(fc_w[:, :expected_fc_in].clone())
        m.fc.in_features = expected_fc_in
    return m
