def shift_counts(n: int) -> tuple[int, int]:
    data = []
    insert_shifts = 0
    for value in range(n):
        insert_shifts += len(data)
        data.insert(0, value)

    data = []
    append_shifts = 0
    for value in range(n):
        append_shifts += 0
        data.append(value)

    return insert_shifts, append_shifts
