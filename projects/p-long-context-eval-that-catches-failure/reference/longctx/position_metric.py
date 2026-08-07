import numpy as np

def compute_position_curve(results):
    pos_map = {}
    for r in results:
        pos = r["item"]["relative_position"]
        pos_map.setdefault(pos, []).append(r["success"])
    curve = {pos: float(np.mean(vals)) for pos, vals in sorted(pos_map.items())}
    return curve
