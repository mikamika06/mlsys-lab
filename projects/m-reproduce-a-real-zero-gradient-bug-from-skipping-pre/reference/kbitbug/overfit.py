import numpy as np

def find_overfitting_step(log_records):
    eval_losses = [r["eval_loss"] for r in log_records]
    steps = [r["step"] for r in log_records]
    min_idx = int(np.argmin(eval_losses))
    return steps[min_idx]
