def select_viable_pairing(pairings, min_acceptance_rate):
    viable = []
    for pair in pairings:
        if float(pair.get("acceptance_rate", 0.0)) >= float(min_acceptance_rate):
            viable.append(pair["name"])
    return viable
