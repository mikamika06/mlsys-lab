import numpy as np
from loss_spike.model import LossModel
from loss_spike.reducer import AllReduceSimulator
from loss_spike.detector import SpikeDetector

def generate_replay_logs():
    return [[float(x) for x in np.linspace(0.1, 1.0, 16)] for _ in range(10)]
