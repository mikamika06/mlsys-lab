"""Target module matcher interface."""

import re


def resolve_by_suffix(named_modules, suffixes):
    """Match submodules ending with given suffixes."""
    raise NotImplementedError


def resolve_by_regex(named_modules, pattern):
    """Match submodules matching regex pattern."""
    raise NotImplementedError
