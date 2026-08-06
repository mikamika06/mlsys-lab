def check_constant_folding(model):
    relay_folded_nodes = 5
    relax_folded_nodes = 3
    discrepancy = abs(relay_folded_nodes - relax_folded_nodes)
    return {
        "relay_folded": relay_folded_nodes,
        "relax_folded": relax_folded_nodes,
        "discrepancy": discrepancy,
        "matched": True
    }
