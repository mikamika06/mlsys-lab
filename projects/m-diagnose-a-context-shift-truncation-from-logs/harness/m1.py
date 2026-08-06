import ref

def check(workdir):
    from diag.logs import analyze_log_truncation
    logs_cases, _, _ = ref.generate_cases()
    correct = 0
    for lines, expected in logs_cases:
        res = analyze_log_truncation(lines)
        if res == expected:
            correct += 1
    match = 1.0 if correct == len(logs_cases) else 0.0
    out = {"diagnosis_match": match}
    if match == 0.0:
        out["_note"] = f"Expected {expected}, got {res}"
    return out
