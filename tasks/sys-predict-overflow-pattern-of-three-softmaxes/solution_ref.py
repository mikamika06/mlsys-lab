import math

import numpy as np


def _naive(z):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ez = np.exp(z)
        return ez / np.sum(ez)


def _lse(z):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = np.max(z)
        ez = np.exp(z - m)
        return ez / np.sum(ez)


def _online(z):
    m = -math.inf
    s = 0.0
    for x in z:
        new_m = max(m, x)
        s = s * math.exp(m - new_m) + math.exp(x - new_m)
        m = new_m
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.exp(z - m) / s


def _overflowed(p):
    return bool(np.any(~np.isfinite(p)))


def classify_softmax_overflow(z) -> tuple:
    """
    Run three softmax implementations on 1-D score vector z and report
    whether each one's output probability vector contains any inf/nan:
      1. naive: exp(z) / sum(exp(z)), no stabilization.
      2. lse: exp(z - max(z)) / sum(exp(z - max(z))).
      3. online: single-pass streaming softmax with a running max/sum,
         rescaling the running sum whenever the running max updates.

    Return (naive_overflow, lse_overflow, online_overflow).
    """
    z = np.asarray(z, dtype=np.float64)
    return (_overflowed(_naive(z)), _overflowed(_lse(z)), _overflowed(_online(z)))
