def extract_exclusion_evidence(engine_logs):
    excluded = []
    for line in engine_logs:
        if "tactic" in line.lower() and "rejected" in line.lower():
            parts = line.split(":")
            excluded.append(parts[0].strip())
    return sorted(excluded)
