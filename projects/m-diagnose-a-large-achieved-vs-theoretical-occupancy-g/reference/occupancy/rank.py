import ref


def rank_configs(configs):
    limits = ref.get_device_limits()
    scored = []
    for i, cfg in enumerate(configs):
        occ = ref.compute_theoretical_occupancy(cfg, limits)
        scored.append((occ, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in scored]
