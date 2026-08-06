"""Diagnostics for unsatisfiable or non-terminating schema constraints."""

from typing import Any, Dict, List, Tuple


def diagnose_schema_deadlock(
    vocab: Dict[int, str],
    eos_token_id: int,
    schema: Dict[str, Any],
    max_depth: int = 100,
) -> Tuple[bool, str]:
    raise NotImplementedError
