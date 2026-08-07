def parse_release_notes(text):
    """Parse raw release note text into structured release items."""
    flags = {}
    configs = {}
    deprecations = set()
    breaking = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("FLAG:"):
            parts = line[5:].strip().split("=")
            if len(parts) == 2:
                flags[parts[0].strip()] = parts[1].strip()
        elif line.startswith("CONFIG:"):
            parts = line[7:].strip().split("=")
            if len(parts) == 2:
                configs[parts[0].strip()] = parts[1].strip()
        elif line.startswith("DEPRECATED:"):
            item = line[11:].strip()
            if item:
                deprecations.add(item)
        elif line.startswith("BREAKING:"):
            item = line[9:].strip()
            if item:
                breaking.add(item)

    return {
        "flags": flags,
        "configs": configs,
        "deprecations": deprecations,
        "breaking": breaking,
    }
