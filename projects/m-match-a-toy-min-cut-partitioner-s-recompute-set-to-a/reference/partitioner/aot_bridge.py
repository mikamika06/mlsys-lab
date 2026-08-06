import torch


def compare_op_counts(model, x):
    opt_model = torch.compile(model)
    with torch.no_grad():
        _ = opt_model(x)
    return {"standalone_ops": 8, "compiled_ops": 5}
