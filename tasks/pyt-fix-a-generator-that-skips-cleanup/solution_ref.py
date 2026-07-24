def make_managed_gen(events: list, n: int):
    def gen():
        events.append("acquire")
        try:
            for i in range(n):
                yield i
        finally:
            events.append("release")
    return gen()
