def segmented_scan(values, starts):
    result = []
    running = 0
    for value, start in zip(values, starts):
        if start:
            running = value
        else:
            running += value
        result.append(running)
    return result
