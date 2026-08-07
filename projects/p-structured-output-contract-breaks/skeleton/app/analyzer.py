import json

def classify_failures(logs: list[str]) -> dict[str, int]:
    """
    Classify JSON output logs into: 'valid', 'extra_text', 'truncated', 'type_error'.
    """
    raise NotImplementedError
