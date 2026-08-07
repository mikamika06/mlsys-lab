M1_IN_RANGE = [
    ({"name": "batch", "min": 1, "max": 128}, 32),
    ({"name": "seq", "min": 1, "max": 1024}, 1024),
    ({"name": "hidden", "min": 256, "max": 256}, 256),
]

M1_OUT_RANGE = [
    ({"name": "batch", "min": 1, "max": 128}, 256),
    ({"name": "seq", "min": 1, "max": 1024}, 2048),
    ({"name": "hidden", "min": 256, "max": 256}, 512),
]


def get_msg(name, val, min_val, max_val):
    return f"Dimension '{name}' got size {val} which is outside range [{min_val}, {max_val}]. Increase max_val if this batch size is expected."
