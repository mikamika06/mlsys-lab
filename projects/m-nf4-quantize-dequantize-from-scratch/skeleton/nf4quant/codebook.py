"""Codebook derivations for NF4, FP4, and INT4 formats."""

import numpy as np


def norm_ppf(p):
    """Acklam's approximation for standard normal quantile function."""
    raise NotImplementedError


def create_nf4_codebook():
    """Derive 16-element NF4 codebook from normal distribution quantiles."""
    raise NotImplementedError


def create_fp4_codebook():
    """Derive 16-element FP4 (E2M1) normalized codebook."""
    raise NotImplementedError


def create_int4_codebook():
    """Derive 16-element uniform INT4 codebook in [-1, 1]."""
    raise NotImplementedError
