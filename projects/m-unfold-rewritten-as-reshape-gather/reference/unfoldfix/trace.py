def minimal_source_trace(tb_text, user_prefix):
    lines = tb_text.split("\n")
    best = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if "File \"" in line and user_prefix in line:
            parts = line.split('"')
            filename = parts[1]
            line_part = line.split(",")[1].strip()
            lineno = int(line_part.split()[1])
            best = {"file": filename, "line": lineno}
        i += 1
    return best
