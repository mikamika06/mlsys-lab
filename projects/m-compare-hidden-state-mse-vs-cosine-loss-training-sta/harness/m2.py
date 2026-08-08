import ref
import numpy as np


def check(workdir):
    from distill.mapping import map_layers, compute_loss_magnitude
    from distill.sim import simulate_stability

    out = {"mapping_match": 0.0, "rel_err": 0.0}
    s_layers = list(range(4))
    t_layers = list(range(8))

    strategies = ["uniform", "top", "other"]
    match_count = 0

    for strat in strategies:
        want_map = ref.ref_map_layers(s_layers, t_layers, strat)
        got_map = map_layers(s_layers, t_layers, strat)
        if got_map == want_map:
            match_count += 1

    s_states = np.random.randn(8, 16, 32)
    t_states = np.random.randn(8, 16, 32)
    mapping = [(0, 0), (1, 2), (2, 4), (3, 6)]

    want_mag = ref.ref_compute_loss_magnitude(s_states, t_states, mapping, "mse")
    got_mag = float(compute_loss_magnitude(s_states, t_states, mapping, "mse"))

    if abs(want_mag - got_mag) < 1e-5:
        match_count += 1

    want_sim = ref.ref_simulate_stability([1.0, 2.0, 15.0, 3.0], threshold=10.0)
    got_sim = simulate_stability([1.0, 2.0, 15.0, 3.0], threshold=10.0)

    if got_sim == want_sim:
        match_count += 1

    out["mapping_match"] = float(match_count)

    if want_mag > 0:
        rel = abs(want_mag - got_mag) / abs(want_mag)
    else:
        rel = 0.0
    out["rel_err"] = float(rel)

    return out
