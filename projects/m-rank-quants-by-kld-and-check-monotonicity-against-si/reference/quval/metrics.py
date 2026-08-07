import numpy as np


def rank_quants_by_kld(quants):
    scored = []
    for q in quants:
        logits_ref = np.array(q["logits_ref"])
        logits_q = np.array(q["logits_q"])
        p = np.exp(logits_ref - np.max(logits_ref, axis=-1, keepdims=True))
        p /= np.sum(p, axis=-1, keepdims=True)
        log_p = np.log(p + 1e-12)

        q_probs = np.exp(logits_q - np.max(logits_q, axis=-1, keepdims=True))
        q_probs /= np.sum(q_probs, axis=-1, keepdims=True)
        log_q = np.log(q_probs + 1e-12)

        kld = np.sum(p * (log_p - log_q), axis=-1).mean()
        scored.append({"name": q["name"], "size_bytes": q["size_bytes"], "kld": float(kld), "ppl": q["ppl"]})

    scored.sort(key=lambda x: x["kld"])
    return scored


def check_monotonicity(ranked_quants):
    violations = 0
    for i in range(len(ranked_quants) - 1):
        if ranked_quants[i]["size_bytes"] > ranked_quants[i + 1]["size_bytes"]:
            if ranked_quants[i]["kld"] > ranked_quants[i + 1]["kld"]:
                violations += 1
    return violations == 0


def find_disagreements(quants):
    ranked_kld = rank_quants_by_kld(quants)
    ranked_ppl = sorted(quants, key=lambda x: x["ppl"])

    kld_order = [q["name"] for q in ranked_kld]
    ppl_order = [q["name"] for q in ranked_ppl]

    disagreements = []
    for name in kld_order:
        if kld_order.index(name) != ppl_order.index(name):
            disagreements.append(name)
    return disagreements
