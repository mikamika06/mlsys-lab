import numpy as np
from evalrec.metrics import compute_perplexity


def detect_contaminated_baseline(
    teacher_data: dict,
    historical_ppl: dict,
    max_perplexity_drop_ratio: float = 0.4,
    min_entropy_threshold: float = 1e-3,
) -> dict:
    flagged = []
    cat_stats = {}

    for cat in sorted(teacher_data.keys()):
        data = teacher_data[cat]
        logits = data["logits"]
        targets = data["targets"]

        ppl = compute_perplexity(logits, targets)

        c = np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = c + np.log(
            np.sum(np.exp(logits - c), axis=-1, keepdims=True)
        )
        log_p = logits - log_sum_exp
        p = np.exp(log_p)

        entropies = -np.sum(p * log_p, axis=-1)
        mean_entropy = float(np.mean(entropies))

        hist_ppl = historical_ppl.get(cat, ppl)
        ppl_drop = ppl < (1.0 - max_perplexity_drop_ratio) * hist_ppl
        low_ent = mean_entropy < min_entropy_threshold

        if ppl_drop and low_ent:
            reason = "both"
        elif ppl_drop:
            reason = "perplexity_drop"
        elif low_ent:
            reason = "low_entropy"
        else:
            reason = "ok"

        is_flagged = ppl_drop or low_ent
        if is_flagged:
            flagged.append(cat)

        cat_stats[cat] = {
            "perplexity": float(ppl),
            "entropy": float(mean_entropy),
            "flagged": is_flagged,
            "reason": reason,
        }

    return {
        "is_contaminated": len(flagged) > 0,
        "flagged_categories": sorted(flagged),
        "category_stats": cat_stats,
    }
