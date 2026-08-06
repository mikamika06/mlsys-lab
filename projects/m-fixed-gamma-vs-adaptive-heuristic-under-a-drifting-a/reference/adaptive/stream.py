import math
import random


def generate_stream(cfg):
    rng = random.Random(cfg["seed"])
    alphas = []
    for t in range(cfg["steps"]):
        if cfg["drift"] == "linear":
            a = max(0.1, min(0.95, cfg["base_alpha"] - 0.5 * (t / cfg["steps"])))
        elif cfg["drift"] == "sine":
            a = max(0.1, min(0.95, cfg["base_alpha"] + 0.3 * math.sin(t * 0.1)))
        else:
            a = 0.2 if (t // 30) % 2 == 0 else 0.8
        alphas.append(round(a, 4))
    return alphas
