def scan_work(n: int, algorithm: str) -> int:
    if n <= 0:
        raise ValueError("non-positive size")

    if algorithm == "hillis_steele":
        count = 0
        distance = 1
        while distance < n:
            count += max(0, n - distance)
            distance *= 2
        return count

    if algorithm == "blelloch":
        size = 1
        while size < n:
            size *= 2

        count = 0
        step = 2
        while step <= size:
            count += size // step
            step *= 2

        step = size
        while step >= 2:
            count += size // step
            step //= 2

        return count

    raise ValueError("unsupported algorithm")
