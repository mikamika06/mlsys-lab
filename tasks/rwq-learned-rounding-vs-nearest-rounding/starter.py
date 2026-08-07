import itertools
import math

def rounding_output_mse(W: list[list[float]], X: list[list[float]], nbits: int) -> tuple[float, float]:
    """
    Compute (mse_learned, mse_rtn) as described in task.md:

    - mse_rtn: mean squared layer-output error using plain nearest-integer
      rounding (RTN) with a per-row symmetric scale.
    - mse_learned: mean squared layer-output error using the best possible
      per-element round-down/round-up choice (brute force over all 2^d_in
      combinations per row), minimizing that row's output error against
      the calibration activations X.
    """
    raise NotImplementedError('your code here')
