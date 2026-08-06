import numpy as np

def get_net_negative_speedup_config() -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    Return (p, q, gamma, t_draft, t_target) such that speedup < 1.0.
    p and q must be valid probability distributions over a vocab of size 2.
    """
    raise NotImplementedError

def prompt_lookup_draft(sequence: list[int], gamma: int) -> list[int]:
    """
    Return up to `gamma` drafted tokens using n-gram prompt lookup.
    Must find the longest suffix of `sequence` that appears earlier in the sequence.
    If there are ties for length, pick the latest occurrence.
    """
    raise NotImplementedError

def get_degenerate_loop_scenario() -> tuple[list[int], int]:
    """
    Return (sequence, gamma) where gamma >= 4 and prompt_lookup_draft
    returns a repeating n-gram loop.
    """
    raise NotImplementedError
