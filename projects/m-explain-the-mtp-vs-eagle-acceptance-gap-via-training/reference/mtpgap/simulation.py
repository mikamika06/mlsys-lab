import numpy as np


def simulate_acceptance_rates(mtp_probs, eagle_probs, temperature):
    t = max(temperature, 1e-5)
    def compute_rate(probs):
        scaled = probs ** (1.0 / t)
        norm = scaled / np.sum(scaled, axis=-1, keepdims=True)
        top1 = np.argmax(norm, axis=-1)
        agreement = np.mean(top1 == np.argmax(probs, axis=-1))
        return float(agreement * 0.8 + 0.2)
    return {
        "mtp_rate": compute_rate(mtp_probs),
        "eagle_rate": compute_rate(eagle_probs)
    }


def compute_trajectory_divergence(seq_mtp, seq_eagle):
    diff = seq_mtp - seq_eagle
    return float(np.mean(np.linalg.norm(diff, axis=-1)))
