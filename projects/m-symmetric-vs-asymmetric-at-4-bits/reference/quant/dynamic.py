import re


def match_dynamic_override(pattern, text):
    try:
        return bool(re.search(pattern, text))
    except Exception:
        return False
