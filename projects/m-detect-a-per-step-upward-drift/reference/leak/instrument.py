def instrument_loop(steps, step_fn, dump_callback=None):
    """Instrument a real loop and dump a snapshot."""
    history = []
    for step in steps:
        res = step_fn(step)
        if dump_callback is not None and step % 2 == 0:
            snap = {"step": step, "allocated": res.get("allocated", 100)}
            dump_callback(snap)
            history.append(snap)
    return history
