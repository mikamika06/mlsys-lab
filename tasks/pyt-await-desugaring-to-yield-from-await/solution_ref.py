def _drive(coro):
    yielded = []
    while True:
        try:
            yielded.append(coro.send(None))
        except StopIteration as exc:
            return yielded, exc.value


def await_desugar_trace(values):
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
