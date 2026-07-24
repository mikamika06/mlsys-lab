def _oracle(coro_factory):
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


class _Awaitable:
    def __init__(self, marker):
        self.marker = marker

    def __await__(self):
        received = yield self.marker
        return received


def _make_factory(markers, final_offset):
    async def run():
        total = 0
        for marker in markers:
            total += await _Awaitable(marker)
        return total + final_offset

    return run


def grade(sol, fx) -> dict:
    cases = [
        _make_factory([3], 5),
        _make_factory([7, 11], 2),
        _make_factory([], 9),
        _make_factory([1, 4, 8, 16], 3),
    ]

    ok = 1.0
    for factory in cases:
        try:
            expected = _oracle(factory)
            got = sol.drive_coroutine(factory)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
