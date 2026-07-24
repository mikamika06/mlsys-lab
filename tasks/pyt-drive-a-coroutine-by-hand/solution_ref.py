def drive_coroutine(coro_factory):
    coro = coro_factory()
    yielded = []
    value = None

    while True:
        try:
            value = coro.send(value)
            yielded.append(value)
            value = 1
        except StopIteration as exc:
            return yielded, exc.value
