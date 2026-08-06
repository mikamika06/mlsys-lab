"""Core ML StateType repair utilities."""

from typing import Dict, List, Tuple


def repair_state_names(
    model_spec: Dict,
    expected_state_names: Tuple[str, str],
) -> Dict:
    """Repair StateType metadata name mismatches in exported Core ML model spec."""
    raise NotImplementedError


def check_state_alignment(
    model_spec: Dict,
    expected_state_names: Tuple[str, str],
) -> bool:
    """Check if model StateType metadata matches the expected state names."""
    raise NotImplementedError
