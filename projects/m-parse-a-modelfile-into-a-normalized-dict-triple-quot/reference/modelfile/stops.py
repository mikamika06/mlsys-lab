import re

def find_missing_stops(ast: dict) -> list:
    template = ast.get("TEMPLATE", "")
    tokens = set(re.findall(r"<\|.*?\|>", template))
    stops = set(ast.get("PARAMETER", {}).get("stop", []))
    return sorted(list(tokens - stops))
