class _RefFuture:
    def __init__(self, delay, value):
        self.delay = delay
        self.value = value
        self.ready_tick = None

    def result(self):
        return self.value


def _ref_run_event_loop(tasks):
    runnable = [(task, None) for task in tasks]
    waiting = []
    done = []
    tick = 0

    while runnable or waiting:
        tick += 1

        still_waiting = []
        for future, task in waiting:
            if future.ready_tick <= tick:
                runnable.append((task, future.result()))
            else:
                still_waiting.append((future, task))
        waiting = still_waiting

        current = runnable
        runnable = []

        for task, value in current:
            try:
                if value is None:
                    yielded = next(task)
                else:
                    yielded = task.send(value)
            except StopIteration as exc:
                done.append(exc.value)
                continue

            if yielded.ready_tick is None:
                yielded.ready_tick = tick + yielded.delay
            waiting.append((yielded, task))

    return done


def _case_specs():
    return [
        [("a", [(1, "x")]), ("b", [(3, "y")]), ("c", [(2, "z")])],
        [
            ("first", [(2, "m"), (1, "n")]),
            ("second", [(1, "p")]),
            ("third", [(4, "q")]),
        ],
        [
            ("left", [(3, "l")]),
            ("middle", [(1, "m"), (3, "n")]),
            ("right", [(2, "r"), (1, "s")]),
        ],
    ]


def _make_tasks(Future, specs):
    def job(name, futures):
        parts = []
        for future in futures:
            parts.append((yield future))
        return name + ":" + ",".join(parts)

    return [
        job(name, [Future(delay, value) for delay, value in futures])
        for name, futures in specs
    ]


def grade(sol, fx) -> dict:
    ok = 1.0

    for specs in _case_specs():
        expected = _ref_run_event_loop(
            _make_tasks(_RefFuture, specs)
        )

        try:
            got = sol.run_event_loop(
                _make_tasks(sol.Future, specs)
            )
        except Exception:
            ok = 0.0
            break

        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
