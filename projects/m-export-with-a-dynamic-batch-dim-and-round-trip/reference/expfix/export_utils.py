from torch.export import export, dynamic_dim

def export_with_dynamic_batch(model, sample_input):
    dim0 = dynamic_dim(sample_input, 0)
    ep = export(model, (sample_input,), dynamic_shapes={"x": {0: dim0}})
    return ep
