import re

def parse_op_histogram(text_dump):
    histogram = {}
    pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\d+)")
    for line in text_dump.splitlines():
        match = pattern.match(line)
        if match:
            op_name, count_str = match.groups()
            histogram[op_name] = int(count_str)
    return histogram
