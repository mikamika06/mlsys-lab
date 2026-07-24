def classify_layouts(line_bytes: int) -> list:
    """Return True/False for each of 5 layouts: True = false-sharing occurs."""
    num_threads = 4

    def layout_addrs(layout_id):
        t_range = list(range(num_threads))
        if layout_id == 0:
            return [t * 8 for t in t_range]
        elif layout_id == 1:
            return [t * 64 for t in t_range]
        elif layout_id == 2:
            return [t * 128 for t in t_range]
        elif layout_id == 3:
            return [t * 8 + 64 * (t % 2) for t in t_range]
        elif layout_id == 4:
            return [t * 16 for t in t_range]

    def has_false_sharing(addrs):
        lines = [a // line_bytes for a in addrs]
        return len(lines) != len(set(lines))

    return [has_false_sharing(layout_addrs(i)) for i in range(5)]
