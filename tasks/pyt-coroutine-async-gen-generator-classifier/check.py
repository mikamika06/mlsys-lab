def _ref(fn):
    flags = fn.__code__.co_flags
    CO_COROUTINE = 0x2000
    CO_ASYNC_GENERATOR = 0x8000
    CO_GENERATOR = 0x20
    return [int(bool(flags & CO_COROUTINE)),
            int(bool(flags & CO_ASYNC_GENERATOR)),
            int(bool(flags & CO_GENERATOR))]

def grade(sol, fx) -> dict:
    import asyncio
    def normal_func(): pass

    async def async_coroutine():
        await asyncio.sleep(0)

    async def async_gen():
        yield 1

    def generator_func():
        yield 1

    cases = [
        (normal_func,),
        (async_coroutine,),
        (async_gen,),
        (generator_func,)
    ]

    ok = 1.0
    for fn_tuple in cases:
        fn = fn_tuple[0]
        try:
            got = sol.classify(fn)
        except Exception:
            ok = 0.0
            break
        exp = _ref(fn)
        if list(got) != exp:
            ok = 0.0
            break
    return {"exact_match": ok}
