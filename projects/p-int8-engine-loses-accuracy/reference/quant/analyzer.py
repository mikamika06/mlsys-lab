import numpy as np


def compare_fp16_int8(fp16_activations, int8_activations):
    results = {}
    for name in fp16_activations:
        ref = np.array(fp16_activations[name], dtype=np.float64)
        quant = np.array(int8_activations[name], dtype=np.float64)

        diff = ref - quant
        mae = float(np.mean(np.abs(diff)))

        ref_var = float(np.var(ref))
        noise_var = float(np.var(diff))
        if noise_var < 1e-12:
            snr = 100.0
        else:
            snr = float(10.0 * np.log10(ref_var / noise_var))

        ref_norm = float(np.linalg.norm(ref))
        quant_norm = float(np.linalg.norm(quant))
        if ref_norm < 1e-12 or quant_norm < 1e-12:
            cos_sim = 1.0
        else:
            cos_sim = float(np.sum(ref * quant) / (ref_norm * quant_norm))

        results[name] = {"snr": snr, "cosine_sim": cos_sim, "mae": mae}
    return results


def rank_sensitive_layers(metrics):
    ranked = []
    for name, m in metrics.items():
        score = m["mae"] - (0.1 * m["snr"])
        ranked.append((name, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
