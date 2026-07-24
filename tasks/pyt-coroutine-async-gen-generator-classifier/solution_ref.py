def classify(fn):
    flags = fn.__code__.co_flags
    CO_COROUTINE = 0x2000
    CO_ASYNC_GENERATOR = 0x8000
    CO_GENERATOR = 0x20
    return [int(bool(flags & CO_COROUTINE)),
            int(bool(flags & CO_ASYNC_GENERATOR)),
            int(bool(flags & CO_GENERATOR))]
