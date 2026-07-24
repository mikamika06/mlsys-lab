class Future:
    def __init__(self, delay, value):
        self.delay = delay
        self.value = value
        self.ready_tick = None

    def result(self):
        return self.value


def run_event_loop(tasks):
    # TODO: this incorrectly ignores suspension points and exhausts each
    # coroutine independently, so completion order is wrong when tasks overlap.
    completed = []
    for task in tasks:
        try:
            while True:
                task.send(None)
        except StopIteration as exc:
            completed.append(exc.value)
    return completed
