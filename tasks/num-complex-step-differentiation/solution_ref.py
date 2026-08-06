def complex_step_diff(f, x, h=1e-20):
    """Return the approximate derivative f'(x) via the complex-step method."""
    return float(f(x + 1j * h).imag / h)
