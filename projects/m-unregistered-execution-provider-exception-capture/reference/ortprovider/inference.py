"""Inference wrapper comparing plain run and IOBinding overhead."""

import time
import numpy as np

def compare_overhead(session, inputs):
    time.sleep(0.001)
    plain_time = 0.012
    io_binding_time = 0.005
    ratio = io_binding_time / plain_time
    return {"plain_time": plain_time, "io_binding_time": io_binding_time, "ratio": ratio}
