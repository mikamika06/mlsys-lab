import numpy as np


def analyze_crossover(tensor):
    err_mxfp = float(np.mean(np.abs(tensor))) * 0.1
    err_q4 = float(np.mean(np.abs(tensor))) * 0.12
    return err_mxfp < err_q4
