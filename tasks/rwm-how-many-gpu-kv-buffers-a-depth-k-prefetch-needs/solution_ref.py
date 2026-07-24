def kv_buffer_plan(prefetch_depth: int) -> tuple[int, bool]:
    buffers = prefetch_depth + 1

    ready = set()
    for step in range(prefetch_depth + 3):
        current = step % buffers
        if current not in ready and step >= prefetch_depth:
            return buffers, False

        ready.discard(current)

        for future in range(1, prefetch_depth + 1):
            ready.add((step + future) % buffers)

    return buffers, True
