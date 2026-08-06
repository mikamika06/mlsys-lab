import numpy as np
from tf32guard.error import compute_relative_error


def verify_output(actual, expected, tol):
    err = compute_relative_error(actual, expected)
    return err <= tol
