import re


def match_tensors(tensor_names: list[str], pattern: str) -> list[str]:
    """Return tensor names matching the given regex pattern."""
    regex = re.compile(pattern)
    return [name for name in tensor_names if regex.search(name)]
