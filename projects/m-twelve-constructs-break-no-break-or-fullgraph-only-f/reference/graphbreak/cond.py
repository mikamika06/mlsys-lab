import torch

def safe_conditional(pred, x, y):
    def true_fn(val_x, val_y):
        return val_x * 2 + val_y
    def false_fn(val_x, val_y):
        return val_x - val_y
    return torch.cond(pred, true_fn, false_fn, (x, y))
