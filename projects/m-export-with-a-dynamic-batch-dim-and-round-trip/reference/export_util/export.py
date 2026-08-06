import torch
from torch.export import export, Dim

def export_with_dynamic_batch(model, sample_inputs):
    batch_dim = Dim("batch", min=1, max=128)
    dynamic_shapes = {"x": {0: batch_dim}}
    ep = export(model, sample_inputs, dynamic_shapes=dynamic_shapes)
    return ep
