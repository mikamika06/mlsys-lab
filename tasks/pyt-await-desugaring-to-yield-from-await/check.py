def _drive(coro):
    yielded = []
    while True:
        try:
            yielded.append(coro.send(None))
        except StopIteration as exc:
            return yielded, exc.value


def _oracle(values):
    class Awaitable:
        def __init__(self, items):
            self.items = list(items)

        def __await__(self):
            for item in self.items:
                yield item
            return "done"

    async def use_await():
        return await Awaitable(values)

    await_yields, await_result = _drive(use_await())

    def use_yield_from():
        result = yield from Awaitable(values).__await__()
        return result

    gen = use_yield_from()
    yield_yields = []
    while True:
        try:
            yield_yields.append(next(gen))
        except StopIteration as exc:
            yield_result = exc.value
            break

    return (await_yields, await_result, yield_yields, yield_result)


def grade(sol, fx) -> dict:
    cases = [
        [],
        ["a"],
        ["x", "y", "z"],
        ["first", "second", "third", "fourth"],
    ]
    ok = 1.0
    for values in cases:
        try:
            got = sol.await_desugar_trace(list(values))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(list(values)):
            ok = 0.0
            break
    return {"exact_match": ok}
