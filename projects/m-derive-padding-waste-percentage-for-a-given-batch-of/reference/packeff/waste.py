def compute_waste_percentage(lengths, max_length):
    if not lengths:
        return 0.0
    total_capacity = len(lengths) * max_length
    total_actual = sum(lengths)
    return ((total_capacity - total_actual) / total_capacity) * 100.0
