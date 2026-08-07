import json
from typing import Any, Dict, List, Tuple


def validate_tool_call(
    tool_call: Dict[str, Any], schemas: Dict[str, Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    raise NotImplementedError


def demonstrate_format_json_non_conformance(
    raw_responses: List[Dict[str, Any]], schema: Dict[str, Any]
) -> List[Dict[str, Any]]:
    raise NotImplementedError
