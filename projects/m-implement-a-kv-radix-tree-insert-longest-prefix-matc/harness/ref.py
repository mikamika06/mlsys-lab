import random

TRACES = []


def generate_traces():
    global TRACES
    if TRACES:
        return TRACES

    rng = random.Random(42)
    system_prompt = [101, 202, 303, 404, 505, 606, 707, 808]
    tool_schemas = [901, 902, 903, 904, 905, 906, 907, 908, 909, 910]

    trace = []
    for turn in range(20):
        user_msg = [rng.randint(1000, 2000) for _ in range(rng.randint(5, 12))]
        if turn % 2 == 0:
            prompt = system_prompt + tool_schemas + user_msg
        else:
            prompt = system_prompt + user_msg + [rng.randint(2000, 3000) for _ in range(4)]
        trace.append({"turn": turn, "tokens": prompt})

    TRACES = [trace]
    return TRACES


generate_traces()
