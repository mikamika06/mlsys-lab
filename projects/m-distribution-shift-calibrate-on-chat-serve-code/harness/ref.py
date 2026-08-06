import numpy as np

np.random.seed(42)
CHAT_ACTS = np.random.normal(loc=1.0, scale=0.2, size=(50, 32))
CODE_ACTS = np.random.normal(loc=1.5, scale=0.5, size=(50, 32))
CHAT_SCALES = np.ones(32)

from calib.shift import compute_shift
from calib.metrics import relative_error
from calib.adjust import adjust_scales

def get_oracle_shift():
    return compute_shift(CHAT_ACTS, CODE_ACTS)

def get_oracle_adjusted():
    s = get_oracle_shift()
    return adjust_scales(CHAT_SCALES, s)
