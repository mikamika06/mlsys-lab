def run_protocol():
    def protocol():
        try:
            value = yield "ready"
            while True:
                try:
                    value = yield "received:" + value
                except ValueError as exc:
                    value = yield "handled:" + str(exc)
        finally:
            return "closed"

    g = protocol()
    events = []
    events.append(("next", next(g)))
    events.append(("send", g.send("alpha")))
    events.append(("throw", g.throw(ValueError("bad"))))
    events.append(("send", g.send("beta")))
    try:
        g.close()
        events.append(("close", "closed"))
    except StopIteration as exc:
        events.append(("close", exc.value))
    return events
