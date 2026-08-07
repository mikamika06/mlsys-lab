def annotate_step(step_data):
    return [f"range:{e}" for e in step_data["events"]]
