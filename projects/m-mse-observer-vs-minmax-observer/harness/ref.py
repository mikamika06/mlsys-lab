import numpy as np

SCHEMES = [
    "int8-sym-tensor",
    "int8-asym-channel",
    "int4-sym-block",
    "int8-sym-channel",
    "int4-asym-tensor"
]

DATASETS = [
    np.array([-2.5, -0.5, 0.1, 1.2, 3.4, 10.0], dtype=np.float32),
    np.array([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float32)
]
