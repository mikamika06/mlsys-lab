def modeled_access_count(m, n, access_pattern, layout):
    """Return the modeled sequential-access count for the given layout."""
    def addr(r, c):
        if layout == "row":
            return r * n + c
        else:
            return c * m + r

    if len(access_pattern) < 2:
        return 0

    addrs = [addr(r, c) for r, c in access_pattern]
    return sum(1 for i in range(len(addrs) - 1)
               if abs(addrs[i + 1] - addrs[i]) == 1)
