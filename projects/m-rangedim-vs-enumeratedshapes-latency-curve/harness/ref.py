import numpy as np

SAMPLE_INPUTS = [16, 32, 48, 64, 96, 128]
SAMPLE_HISTOGRAM = {16: 10, 24: 15, 32: 30, 64: 25, 96: 10, 128: 5}
SAMPLE_RANGES = [16, 32, 64, 128]
SAMPLE_PROFILE = {
    ("rangedim", 16): 1.4, ("enumerated", 16): 1.0,
    ("rangedim", 32): 2.2, ("enumerated", 32): 1.5,
    ("rangedim", 64): 3.8, ("enumerated", 64): 2.5,
    ("rangedim", 128): 6.0, ("enumerated", 128): 4.0
}
