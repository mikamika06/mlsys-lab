import time
import math
import numpy as np
import torch


def benchmark_step(fn, is_cuda=False, warmup=10, reps=100):
    raise NotImplementedError


def compute_required_reps(times_sample, tolerance=0.05, confidence_z=1.96):
    raise NotImplementedError
