def merge_partials(partials: list[list[float]]) -> tuple[list[float], list[float]]:
    if not partials:
        return [], []

    indexed = sorted(enumerate(partials), key=lambda item: item[0])
    ordered = [x for _, x in indexed]

    num_rows = len(ordered)
    num_cols = len(ordered[0])

    merged_list = []
    for j in range(num_cols):
        col_sum = 0.0
        for i in range(num_rows):
            col_sum += float(ordered[i][j])
        merged_list.append(col_sum)

    return list(merged_list), list(merged_list)
