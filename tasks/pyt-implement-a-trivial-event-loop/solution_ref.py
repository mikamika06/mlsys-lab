class Future:
    def __init__(self, delay, value):
        self.delay = delay
        self.value = value
        self.ready_tick = None

    def result(self):
        return self.value


def run_event_loop(tasks):
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
