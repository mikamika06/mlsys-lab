import re

UNSUPPORTED_FILTERS = {"tojson", "regex_replace", "fromjson", "groupby", "sort"}

def find_unsupported_filters(template_str):
    found = set()
    pattern = r'\|\s*([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(pattern, template_str)
    for m in matches:
        if m in UNSUPPORTED_FILTERS:
            found.add(m)
    return sorted(list(found))
