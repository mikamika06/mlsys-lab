import re

def extract_program_id(error_log: str) -> tuple:
    # Matches: program_id (1, 2, 3)
    pattern = re.compile(r"program_id\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
    match = pattern.search(error_log)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return ()
