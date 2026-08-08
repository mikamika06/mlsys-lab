import sys
import numpy as np

sys.path.insert(0, ".")
from reduction.variance import one_pass_variance, two_pass_variance

def test_variance_stability():
    """
    Write a test that proves two_pass_variance is more accurate than one_pass_variance
    on data prone to catastrophic cancellation in float16.
    """
    raise NotImplementedError
