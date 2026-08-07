import numpy as np


def compute_max_scale(w_chan, qmax=7.0):
    raise NotImplementedError


def simulate_quant(w_chan, scale, qmin=-8, qmax=7):
    raise NotImplementedError


def find_best_scale_mse(w_chan, num_candidates=100, qmin=-8, qmax=7):
    raise NotImplementedError
