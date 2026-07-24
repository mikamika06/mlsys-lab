def make_managed_gen(events: list, n: int):
    """Return a generator that appends "acquire" to events on first resume,
    yields 0..n-1, and appends "release" to events exactly once no matter
    whether it's exhausted, closed early, or garbage collected early.
    """
    def gen():
        events.append("acquire")
        # BUG: this cleanup line only runs if the loop finishes naturally.
        # An early .close() or garbage collection throws GeneratorExit in
        # at the suspended yield, which unwinds straight out of the
        # function -- skipping this line entirely.
        for i in range(n):
            yield i
        events.append("release")
    return gen()
