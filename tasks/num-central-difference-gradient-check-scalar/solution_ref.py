from __future__ import annotations

from typing import Callable


def central_diff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Central finite-difference estimate of f'(x)."""
    x = float(x)
    h = float(h)
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def grad_check(f: Callable[[float], float], grad_f: Callable[[float], float],
               x: float, h: float = 1e-5) -> float:
    """Symmetric relative disagreement between the numeric and analytic gradient."""
    num = central_diff(f, x, h)
    ana = float(grad_f(x))
    return float(abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
