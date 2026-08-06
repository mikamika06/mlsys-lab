def parse_argv(argv):
    args = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            if "=" in token:
                key, val = token[2:].split("=", 1)
                args[key] = val
            else:
                key = token[2:]
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    args[key] = argv[i + 1]
                    i += 1
                else:
                    args[key] = True
        i += 1
    return args
