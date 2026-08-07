import numpy as np
from metal_op.kernel import run_custom_kernel


class FusedModel:
    def __init__(self):
        self.scale = 1.0

    def forward(self, x):
        return run_custom_kernel(x) * self.scale
