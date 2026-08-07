import numpy as np

def compute_dataloader_fraction(step_durations, dataloader_waits):
    return float(np.sum(dataloader_waits) / np.sum(step_durations))
