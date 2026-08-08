import ref

def check(workdir):
    from amxlog.parser import parse_verbose_line
    lines = ref.generate_log_lines()
    expected = ref.generate_expected_parses()
    matched = 0
    for line, exp in zip(lines, expected):
        got = parse_verbose_line(line)
        if got == exp:
            matched += 1
    return {"parsed_match": float(matched), "total": float(len(expected))}
