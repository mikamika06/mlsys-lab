from __future__ import annotations

from typing import Callable


def central_diff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Return the central finite-difference estimate of f'(x) with step h."""
    raise NotImplementedError('your code here')


def grad_check(f: Callable[[float], float], grad_f: Callable[[float], float],
               x: float, h: float = 1e-5) -> float:
    """Return |num - ana| / max(|num| + |ana|, 1e-12), where num is the central
    difference of f at x and ana is grad_f(x)."""
    raise NotImplementedError('your code here')
