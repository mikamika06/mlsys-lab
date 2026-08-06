import numpy as np
from mtpgap.model import compute_mtp_loss, compute_eagle_loss
from mtpgap.simulation import simulate_acceptance_rates


def analyze_gap(config):
    logits = config["logits"]
    targets = config["targets"]
    weights = config["weights"]
    mtp_p = config["mtp_probs"]
    eagle_p = config["eagle_probs"]
    temp = config["temperature"]

    mtp_l = compute_mtp_loss(logits, targets, weights)
    eagle_l = compute_eagle_loss(logits[0], targets[0])
    rates = simulate_acceptance_rates(mtp_p, eagle_p, temp)

    return {
        "mtp_loss": mtp_l,
        "eagle_loss": eagle_l,
        "acceptance_rates": rates,
        "gap": rates["eagle_rate"] - rates["mtp_rate"]
    }
