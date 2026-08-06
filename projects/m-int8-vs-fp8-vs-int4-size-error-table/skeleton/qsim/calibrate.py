import numpy as np

def calibrate_scales(acts, target_max=127.0):
    raise NotImplementedError

def compare_domains(act_in, act_out, target_max=127.0):
    raise NotImplementedError

def detect_poison(acts, threshold_ratio=10.0):
    raise NotImplementedError
