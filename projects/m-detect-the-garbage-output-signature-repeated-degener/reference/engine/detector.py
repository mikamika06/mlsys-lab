def detect_garbage(tokens):
    if not tokens:
        return False
    n = len(tokens)
    if n < 4:
        return False
    excl_count = sum(1 for t in tokens if t == "!")
    if excl_count / n > 0.6:
        return True
    for length in range(1, 5):
        if n >= length * 3:
            sub = tokens[-length:]
            matches = 0
            for i in range(1, 4):
                if tokens[-length * (i + 1):-length * i if i > 1 else None] == sub:
                    matches += 1
            if matches >= 2:
                return True
    return False
