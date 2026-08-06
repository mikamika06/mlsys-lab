import numpy as np


def predict_oom_step(steps, mems, capacity):
    x = np.array(steps, dtype=float)
    y = np.array(mems, dtype=float)
    poly = np.polyfit(x, y, 1)
    step = (capacity - poly[1]) / poly[0]
    return int(np.round(step))
