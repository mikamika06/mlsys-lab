def count_recompilations(scenario):
    adapters = scenario["adapters"]
    shared = scenario["shared_base"]
    if shared:
        seen = set()
        recomps = 0
        for ad in adapters:
            if ad not in seen:
                seen.add(ad)
                recomps += 1
        return recomps
    else:
        return len(adapters)
